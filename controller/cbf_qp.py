"""Shared CBF-QP execution filter with OSQP and batched JAX backends."""
from typing import NamedTuple
from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import osqp
from scipy import sparse

from env.robots.base import Obstacles, check_obstacle_shapes
from env.robots import (DoubleIntegratorRobot, HumanoidRobot, QuadrotorRobot,
                        SingleIntegratorRobot, UnicycleRobot)
from solver.jax_qpth.qp import (solve_qp as solve_jax_qp, qp_residuals,
                               check_qp_residuals)


class _TrackingQP(NamedTuple):
    q: object
    p: object
    g: object
    h: object


def _assemble_qp(nominal, A, b, low, high, weights, output_matrix, *,
                         soft_indices, slack_weight, xp):
    """Build 0.5*z.T@q@z+p.T@z subject to g@z<=h.

    The physical variable is z=[u, delta]. Only soft_indices receive nonnegative
    slacks; all other constraints and input bounds stay hard. Leading dimensions
    are batches. Backend-specific conditioning and solving remain with callers.
    """
    nominal = xp.asarray(nominal)
    dtype, leading = nominal.dtype, nominal.shape[:-1]
    dim, rows = nominal.shape[-1], A.shape[-2]
    indices = xp.asarray(soft_indices, dtype=int)
    count = indices.shape[0]
    zeros = lambda *shape: xp.zeros(leading+shape, dtype=dtype)
    batch = lambda value: xp.broadcast_to(value, leading+value.shape)
    matrix = xp.asarray(output_matrix, dtype=dtype)
    weights = xp.asarray(weights, dtype=dtype)
    tracking = xp.swapaxes(matrix, -1, -2) @ (weights[..., :, None]*matrix)
    tracking = xp.broadcast_to(tracking, leading+(dim, dim))
    slack_eye = batch(xp.eye(count, dtype=dtype))
    q = xp.concatenate([
        xp.concatenate([tracking, zeros(dim, count)], axis=-1),
        xp.concatenate([zeros(count, dim), slack_weight*slack_eye], axis=-1),
    ], axis=-2)
    p = xp.concatenate([-(tracking @ nominal[..., None])[..., 0], zeros(count)], axis=-1)
    soft_columns = batch(xp.eye(rows, dtype=dtype)[:, indices])
    constraints = xp.concatenate([-xp.asarray(A, dtype=dtype), -soft_columns], axis=-1)
    eye = batch(xp.eye(dim, dtype=dtype))
    lower = xp.concatenate([-eye, zeros(dim, count)], axis=-1)
    upper = xp.concatenate([eye, zeros(dim, count)], axis=-1)
    nonnegative = xp.concatenate([zeros(count, dim), -slack_eye], axis=-1)
    g = xp.concatenate([constraints, lower, upper, nonnegative], axis=-2)
    low = xp.broadcast_to(xp.asarray(low, dtype=dtype), leading+(dim,))
    high = xp.broadcast_to(xp.asarray(high, dtype=dtype), leading+(dim,))
    h = xp.concatenate([xp.asarray(b, dtype=dtype), -low, high, zeros(count)], axis=-1)
    return _TrackingQP(q, p, g, h)


def _nearest_obstacle_ids(position, obstacles, top_k):
    distance = jnp.linalg.norm(obstacles.centers-position[..., None, :], axis=-1)
    count = obstacles.mask.shape[-1] if top_k is None else min(top_k, obstacles.mask.shape[-1])
    return jax.lax.top_k(-jnp.where(obstacles.mask > .5, distance, jnp.inf), count)[1]


def _obstacle_clearance(robot, state, obstacles):
    """Surface clearance in the geometry used by the robot's distance CBF."""
    position = state[..., :2]
    radius = robot.radius + obstacles.radii.astype(state.dtype) + robot.margin
    if isinstance(robot, UnicycleRobot):
        heading = jnp.stack((jnp.cos(state[..., 2]), jnp.sin(state[..., 2])), axis=-1)
        position = position + robot.lookahead * heading
        radius = radius + robot.lookahead
    return jnp.linalg.norm(position[..., None, :] - obstacles.centers, axis=-1) - radius


def select_obstacles(position, obstacles, top_k):
    """Select nearest obstacles without changing the single-scene array shapes."""
    if top_k is None:
        return obstacles
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("controller.qp_obs_top_k must be null or a positive integer")
    indices = _nearest_obstacle_ids(position, obstacles, top_k)
    mask = jnp.zeros_like(obstacles.mask).at[indices].set(obstacles.mask[indices])
    return obstacles._replace(mask=mask)


@dataclass(frozen=True)
class SolverConfig:
    eps_abs: float = 1e-9
    eps_rel: float = 1e-9
    max_iter: int = 10000
    adaptive_rho_interval: int = 25
    feasibility_tolerance: float = 1e-6
    polishing: bool = False
    verbose: bool = False

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in
                    [self.eps_abs, self.eps_rel, self.feasibility_tolerance])
                or not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool)
                or self.max_iter <= 0
                or not isinstance(self.adaptive_rho_interval, int)
                or isinstance(self.adaptive_rho_interval, bool) or self.adaptive_rho_interval <= 0
                or not isinstance(self.polishing, bool) or not isinstance(self.verbose, bool)):
            raise ValueError("invalid OSQP tolerances, iteration limit or logging settings")


class ControlResult(NamedTuple):
    action: jax.Array          # NaN on primal infeasibility.
    candidate: jax.Array
    residual: jax.Array        # Unrelaxed CBF/input residuals.
    kkt_residual: jax.Array    # Primal, dual and complementarity residuals.
    valid: bool               # QP feasibility and KKT checks.
    initial_safe: bool         # h/psi initial conditions hold.
    correction_norm: float
    solver_status: str
    iterations: int


def _prepare_osqp_data(nominal, constraints, low, high, weights, output_matrix, *,
                      soft_indices=(), slack_weight=0.0):
    """Validate and scale the physical tracking QP."""
    if not jax.config.x64_enabled:
        raise ValueError("CBF-QP requires JAX_ENABLE_X64=true before importing JAX")
    nominal, low, high, weights = [np.asarray(x, dtype=np.float64)
                                   for x in (nominal, low, high, weights)]
    if nominal.ndim != 1 or nominal.size == 0:
        raise ValueError("nominal must be a nonempty vector")
    d = nominal.shape[0]
    if low.shape != (d,) or high.shape != (d,):
        raise ValueError("nominal and bounds must be vectors of equal size")
    scale = np.maximum(np.maximum(np.abs(low), np.abs(high)), 1e-6)
    M = np.asarray(output_matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] != d or weights.shape != (M.shape[0],):
        raise ValueError("output matrix must have D columns and one weight per row")
    cbf_A, cbf_b = np.asarray(constraints.A, dtype=np.float64), np.asarray(constraints.b, dtype=np.float64)
    if cbf_A.ndim != 2 or cbf_A.shape[1] != d or cbf_b.shape != (cbf_A.shape[0],):
        raise ValueError("expected constraint matrix [M,D] and offset [M]")
    domain = np.asarray(constraints.domain, dtype=np.float64)
    if (not all(np.isfinite(x).all() for x in (nominal, low, high, weights, M, cbf_A, cbf_b, domain))
            or np.any(weights <= 0) or np.any(low > high)):
        raise ValueError("QP data must be finite, with positive weights and ordered bounds")
    data = _assemble_qp(nominal, cbf_A, cbf_b, low, high, weights, M,
                                soft_indices=soft_indices, slack_weight=slack_weight, xp=np)
    count = len(soft_indices)
    if count:
        scale = np.concatenate([scale, np.full(count, 1./math.sqrt(slack_weight))])
        nominal = np.concatenate([nominal, np.zeros(count)])
    return nominal, data, scale


def _solve_osqp(nominal, data, scale, domain, solver):
    """Condition and solve the shared physical QP with CPU OSQP."""
    A, b = -data.g, data.h
    P = scale[:, None]*data.q*scale[None, :]
    q = data.p*scale
    tolerance = solver.feasibility_tolerance
    d = nominal.size
    try:
        np.linalg.cholesky(P)
    except np.linalg.LinAlgError as error:
        raise ValueError("tracking objective must determine every control direction") from error
    scaled_A = A * scale[None, :]
    row_scale = np.maximum(np.linalg.norm(scaled_A, axis=1), 1.0)
    C = scaled_A / row_scale[:, None]
    lower = -b / row_scale
    problem = osqp.OSQP()
    try:
        problem.setup(P=sparse.csc_matrix(np.triu(P)), q=q,
                      A=sparse.csc_matrix(C), l=lower, u=np.full(len(b), np.inf),
                      eps_abs=solver.eps_abs, eps_rel=solver.eps_rel,
                      max_iter=solver.max_iter,
                      adaptive_rho_interval=solver.adaptive_rho_interval,
                      polishing=solver.polishing,
                      verbose=solver.verbose)
        result = problem.solve(raise_error=False)
    except osqp.OSQPException as error:
        raise RuntimeError(f"OSQP setup/solve failed: {error}") from error
    status = str(result.info.status)
    if status not in {"solved", "primal infeasible"}:
        raise RuntimeError(f"OSQP failed: status={status}, iterations={result.info.iter}")
    if status == "solved" and (result.x is None or result.y is None):
        raise RuntimeError("OSQP reported solved without a primal/dual solution")
    if status == "solved" and (
            np.shape(result.x) != (d,) or np.shape(result.y) != (len(b),)
            or not np.isfinite(result.x).all() or not np.isfinite(result.y).all()):
        raise RuntimeError("OSQP returned a malformed or nonfinite primal/dual solution")
    candidate = (np.full(d, np.nan) if result.x is None
                 else np.asarray(result.x, dtype=np.float64) * scale)
    residual = A @ candidate + b
    kkt = np.full(3, np.inf)
    if status == "solved":
        # OSQP lower-bound multipliers have the opposite sign to G*x <= h.
        errors = qp_residuals(data.q, data.p, data.g, data.h, candidate,
                              -np.asarray(result.y)/row_scale, xp=np)
        kkt = np.array([errors[0], errors[1], max(errors[2], errors[3])])
    valid = bool(status == "solved" and np.isfinite(kkt).all()
                 and np.all(kkt <= tolerance))
    if status == "solved" and not valid:
        raise RuntimeError(
            f"OSQP solution failed feasibility/KKT checks after {result.info.iter} iterations: "
            f"min_residual={np.min(residual)}, kkt_residual={kkt.tolist()}")
    domain = np.asarray(domain)
    initial_safe = bool(np.isfinite(domain).all() and np.all(domain >= -tolerance))
    return ControlResult(jnp.asarray(candidate if valid else np.full(d, np.nan)),
                        jnp.asarray(candidate), jnp.asarray(residual), jnp.asarray(kkt),
                        valid, initial_safe, float(np.linalg.norm(candidate-nominal)),
                        status, int(result.info.iter))


def validate_slack_weight(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("controller.slack_weight must be finite and nonnegative")


def solve_qp(nominal, constraints, low, high, weights, *, output_matrix,
             solver=SolverConfig(), slack_weight=0.0, soft_mask=None):
    """Solve the CPU tracking QP and return (ControlResult, per-row slack).

    Zero slack_weight keeps every constraint hard. A positive weight requires
    soft_mask and relaxes only its selected rows. Residuals exclude slack;
    validity checks the relaxed QP.
    """
    validate_slack_weight(slack_weight)
    n, d = np.shape(constraints.A)
    ids = ()
    if slack_weight > 0:
        mask = np.asarray(soft_mask)
        if mask.shape != (n,) or not np.isfinite(mask).all():
            raise ValueError("soft_mask must be a finite vector with one entry per CBF row")
        ids = np.flatnonzero(mask > .5)
    augmented_nominal, data, scale = _prepare_osqp_data(
        nominal, constraints, low, high, weights, output_matrix,
        soft_indices=ids, slack_weight=slack_weight)
    augmented = _solve_osqp(augmented_nominal, data, scale, constraints.domain, solver)
    if slack_weight == 0:
        return augmented, jnp.zeros(n, dtype=augmented.candidate.dtype)
    candidate = augmented.candidate[:d]
    raw = jnp.concatenate([
        jnp.asarray(constraints.A)@candidate+jnp.asarray(constraints.b),
        candidate-jnp.asarray(low), jnp.asarray(high)-candidate])
    result = augmented._replace(
        action=augmented.action[:d], candidate=candidate, residual=raw,
        correction_norm=float(np.linalg.norm(np.asarray(candidate)-np.asarray(nominal))))
    slack = jnp.zeros(n, dtype=candidate.dtype).at[ids].set(augmented.candidate[d:])
    return result, slack


@partial(jax.jit, static_argnames=("robot",))
def _control_output_matrix(robot, state):
    return robot.control_output_matrix(state)


@partial(jax.jit, static_argnames=("robot", "speed_facets", "constraint_mode"))
def _controller_constraints(robot, state, obstacles, dt, speed_facets=16, constraint_mode="cbf"):
    kwargs = dict(speed_facets=speed_facets) if isinstance(robot, SingleIntegratorRobot) else {}
    if constraint_mode == "physical":
        return robot.physical_constraints(state, dt, **kwargs)
    return robot.control_constraints(state, obstacles, dt, **kwargs)


def control_residual(robot, state, action, obstacles, dt):
    """Check CBF and input residuals for the applied action."""
    constraints = _controller_constraints(robot, state, obstacles, dt)
    low, high = robot.input_bounds()
    return jnp.concatenate([constraints.A@action+constraints.b, action-low, high-action])


def _validate_controller_inputs(robot, state, nominal, obstacles, dt):
    """Validate shapes and finite values; unsafe states are allowed."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if np.shape(state) != (robot.state_dim,):
        raise ValueError(f"state must have shape ({robot.state_dim},)")
    if np.shape(nominal) != (robot.control_dim,):
        raise ValueError(f"nominal must have shape ({robot.control_dim},)")
    if np.ndim(obstacles.centers) != 2:
        raise ValueError("obstacle centers must have shape [N,2]")
    check_obstacle_shapes(obstacles)
    if not all(np.isfinite(np.asarray(value)).all() for value in (state, nominal, *obstacles)):
        raise ValueError("state, nominal and obstacle data must be finite")
    if np.any(np.asarray(obstacles.radii) < 0):
        raise ValueError("obstacle radii must be nonnegative")


def warmup_controller(robot, state, nominal, obstacles, dt, *, qp_obs_top_k=None):
    """Compile CBF/output computations without solving a QP."""
    _validate_controller_inputs(robot, state, nominal, obstacles, dt)
    obstacles = select_obstacles(state[:2], obstacles, qp_obs_top_k)
    robot.validate_cbf_domain(state, obstacles)
    jax.block_until_ready((_controller_constraints(robot, state, obstacles, dt),
                           _control_output_matrix(robot, state)))


def _prepare_jax_data(robot, state, nominal, obstacles, dt, *, qp_obs_top_k=None,
                 slack_weight=0.0, speed_facets=16, constraint_mode="cbf",
                 single_obstacle=False):
    """Assemble one fixed-CBF QP; whiten variables for mixed input units."""
    x = state.astype(jnp.float64)
    if constraint_mode == "physical":
        ids = jnp.empty((state.shape[0], 0), jnp.int32)
        clearance = jnp.empty((state.shape[0], 0), jnp.float64)
        constraints = jax.vmap(lambda x: _controller_constraints(
            robot, x, None, dt, speed_facets, constraint_mode))(x)
    else:
        if single_obstacle and obstacles.mask.shape[-1]:
            clearance = _obstacle_clearance(robot, x, obstacles)
            # Search all obstacles, without nearest-center top-k truncation.
            # At equal clearance, argmin selects the first obstacle.
            ids = jnp.argmin(jnp.where(obstacles.mask > .5, clearance, jnp.inf),
                             axis=-1, keepdims=True)
        else:
            ids = _nearest_obstacle_ids(state[..., :2], obstacles, qp_obs_top_k)
        batch = jnp.arange(state.shape[0])[:, None]
        obstacles = jax.tree_util.tree_map(lambda value: value[batch, ids], obstacles)
        obstacles = jax.tree_util.tree_map(lambda value: value.astype(jnp.float64), obstacles)
        clearance = _obstacle_clearance(robot, x, obstacles)
        constraints = jax.vmap(lambda x, obs: _controller_constraints(
            robot, x, obs, dt, speed_facets))(x, obstacles)
    dim = robot.control_dim
    slack_dim = obstacles.mask.shape[-1] if slack_weight > 0 else 0
    matrices = jax.vmap(robot.control_output_matrix)(x)
    low, high = robot.input_bounds()
    q, p, g, h = _assemble_qp(
        nominal.astype(jnp.float64), constraints.A, constraints.b, low, high,
        jnp.asarray(robot.weights, jnp.float64), matrices,
        soft_indices=jnp.arange(slack_dim), slack_weight=slack_weight, xp=jnp)
    total = dim+slack_dim
    transform = jsp_linalg.solve_triangular(
        jnp.swapaxes(jnp.linalg.cholesky(q), 1, 2),
        jnp.broadcast_to(jnp.eye(total), q.shape), lower=False)
    scaled_q = jnp.swapaxes(transform, 1, 2) @ q @ transform
    scaled_p = (jnp.swapaxes(transform, 1, 2) @ p[..., None]).squeeze(-1)
    scaled_g = g @ transform
    scale = jnp.maximum(jnp.linalg.norm(scaled_g, axis=-1), 1.)
    return dict(q=scaled_q, p=scaled_p, g=scaled_g/scale[..., None], h=h/scale,
                transform=transform, row_scale=scale, raw_q=q, raw_p=p,
                raw_g=g, raw_h=h, constraints=constraints,
                ids=ids, mask=obstacles.mask, clearance=clearance)


def generate_action(robot, state, nominal, obstacles, dt, *, backend="osqp",
                    solver=SolverConfig(), slack_weight=0.0, qp_obs_top_k=None,
                    speed_facets=16, max_iter=40, eps=1.0e-12, active=None,
                    constraint_mode="cbf", single_obstacle=False):
    """Filter a nominal physical input using OSQP or batched JAX forward solving.

    Only obstacle constraints receive slack. The JAX backend stops gradients and
    accepts single states, batches and active masks; OSQP handles one state.
    Physical mode solves only the existing physical CBFs and input limits once.
    Discrete state-limit violations are reported by the environment.
    single_obstacle selects minimum surface clearance over all active obstacles
    for CBF-RL; the selected robot CBF/HOCBF and physical/input limits are retained.
    """
    if backend not in {"osqp", "jax"}:
        raise ValueError("generate_action backend must be 'osqp' or 'jax'")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("generate_action dt must be positive")
    validate_slack_weight(slack_weight)
    if qp_obs_top_k is not None and (not isinstance(qp_obs_top_k, int)
            or isinstance(qp_obs_top_k, bool) or qp_obs_top_k <= 0):
        raise ValueError("generate_action qp_obs_top_k must be null or a positive integer")
    for name, value, minimum in (("speed_facets", speed_facets, 4), ("max_iter", max_iter, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"generate_action {name} must be an integer >= {minimum}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("generate_action eps must be finite and positive")
    if constraint_mode not in {"cbf", "physical"}:
        raise ValueError("constraint_mode must be 'cbf' or 'physical'")
    if not isinstance(single_obstacle, bool):
        raise ValueError("single_obstacle must be true or false")
    if single_obstacle and (backend != "jax" or constraint_mode != "cbf"):
        raise ValueError("single_obstacle requires the JAX obstacle filter")
    if constraint_mode == "physical":
        if backend != "jax" or slack_weight != 0.:
            raise ValueError("physical-only projection requires the JAX backend and no slack")
    if backend == "osqp":
        if active is not None:
            raise ValueError("active masks require the batched JAX backend")
        _validate_controller_inputs(robot, state, nominal, obstacles, dt)
        obstacles = select_obstacles(state[:2], obstacles, qp_obs_top_k)
        robot.validate_cbf_domain(state, obstacles)
        low, high = robot.input_bounds()
        constraints = _controller_constraints(robot, state, obstacles, dt, speed_facets)
        # Leading obstacle rows may be softened; appended physical limits stay hard.
        soft_mask = (jnp.pad(obstacles.mask, (0, constraints.A.shape[0]-obstacles.mask.size))
                     if slack_weight > 0 else None)
        result, slack = solve_qp(
            nominal, constraints, low, high, jnp.asarray(robot.weights),
            output_matrix=_control_output_matrix(robot, state), solver=solver,
            slack_weight=slack_weight, soft_mask=soft_mask)
        # An infeasibility certificate is not a slack solution for the held input.
        slack = (slack[:obstacles.mask.size] if slack_weight > 0 and result.valid
                 else jnp.zeros_like(obstacles.mask))
        return result.action, {"nominal": nominal, "qp": result, "slack": slack,
                               "obstacle_mask": obstacles.mask}

    single = state.ndim == 1
    expected = state.shape[:-1]+(robot.control_dim,)
    if (state.ndim not in (1, 2) or state.shape[-1] != robot.state_dim
            or nominal.shape != expected):
        raise ValueError("nominal action must match the robot-state batch and Robot.control_dim")
    if single:
        state, nominal = state[None], nominal[None]
        if constraint_mode == "cbf":
            obstacles = jax.tree_util.tree_map(lambda value: value[None], obstacles)
        if active is not None:
            active = jnp.atleast_1d(active)
    if constraint_mode == "physical":
        empty = jnp.empty((state.shape[0], 0), state.dtype)
        vectors = jnp.empty((state.shape[0], 0, 2), state.dtype)
        obstacles = Obstacles(vectors, vectors, empty, empty)
        if active is not None and active.shape != state.shape[:1]:
            raise ValueError("active must match the robot-state batch")
    state, nominal, obstacles = jax.tree_util.tree_map(
        jax.lax.stop_gradient, (state, nominal, obstacles))
    if active is not None:
        initial = robot.initial_state(jnp.zeros(3, state.dtype))
        state = jnp.where(active[:, None], state, initial)
        obstacles = Obstacles(
            jnp.where(active[:, None, None], obstacles.centers, 0.),
            jnp.where(active[:, None, None], obstacles.velocities, 0.),
            jnp.where(active[:, None], obstacles.radii, 0.),
            jnp.where(active[:, None], obstacles.mask, 0.))
        nominal = jnp.where(active[:, None], nominal, 0.)
    data = _prepare_jax_data(robot, state, nominal, obstacles, dt,
                        qp_obs_top_k=qp_obs_top_k, slack_weight=slack_weight,
                        speed_facets=speed_facets, constraint_mode=constraint_mode,
                        single_obstacle=single_obstacle)
    empty = jnp.empty((0,), jnp.float64)
    # Match BarrierNet's first-order versus high-order solver settings.
    high_order = constraint_mode == "physical" or isinstance(robot, (DoubleIntegratorRobot, HumanoidRobot, QuadrotorRobot))
    solution, _, dual, solver_slack = solve_jax_qp(
        data['q'], data['p'], data['g'], data['h'], empty, empty,
        eps=eps, verbose=-1, notImprovedLim=max_iter if high_order else 3,
        maxIter=max_iter, check_Q_spd=False, safe_step=high_order)
    physical = (data['transform'] @ solution[..., None]).squeeze(-1)
    dim = robot.control_dim
    action = physical[:, :dim].astype(nominal.dtype)
    slack = jnp.maximum(physical[:, dim:], 0.)
    if slack_weight == 0:
        slack = jnp.zeros_like(data['mask'])
    errors = qp_residuals(data['raw_q'], data['raw_p'], data['raw_g'], data['raw_h'],
                          physical, dual/data['row_scale'],
                          slack=solver_slack*data['row_scale'])
    finite_inputs = jnp.all(jnp.isfinite(state), axis=1) & jnp.all(jnp.isfinite(nominal), axis=1)
    errors = jnp.where(finite_inputs[:, None], errors, jnp.inf)
    check_qp_residuals(errors, tolerance=solver.feasibility_tolerance,
                       label="Physical CBF-QP" if constraint_mode == "physical" else "SF-PPO CBF-QP")
    primal = errors[:, 0]
    residual = (data['constraints'].A @ action.astype(jnp.float64)[..., None]).squeeze(-1)+data['constraints'].b
    correction = jnp.linalg.norm(action-nominal, axis=1)
    if constraint_mode == "physical":
        diagnostics = {
            'qp_ids': data['ids'],
            'filter/correction': correction,
            'filter/intervention': (correction > 1e-5).astype(jnp.float32),
            'filter/primal_residual': primal,
            'filter/kkt_residual': jnp.max(errors[:, 1:], axis=1),
            'filter/physical_cbf_violation': jnp.max(-residual, axis=1, initial=0.),
            'filter/initial_safe': jnp.all(data['constraints'].domain >= -1e-5, axis=1).astype(jnp.float32),
        }
    else:
        count = jnp.maximum(data['mask'].sum(axis=1), 1.)
        diagnostics = {
            'qp_ids': data['ids'], 'slack': slack, 'cbf_residual': residual[:, :data['mask'].shape[1]],
            'proposed_cbf_residual': ((data['constraints'].A[:, :data['mask'].shape[1]]
                @ nominal.astype(jnp.float64)[..., None]).squeeze(-1)
                + data['constraints'].b[:, :data['mask'].shape[1]]),
            'obstacle_mask': data['mask'],
            'obstacle_clearance': data['clearance'],
            'filter/slack': (slack*data['mask']).sum(axis=1)/count,
            'filter/slack_max': jnp.max(slack*data['mask'], axis=1, initial=0.),
            'filter/correction': correction,
            # Tolerances below affect diagnostic labels only.
            'filter/intervention': (correction > 1e-5).astype(jnp.float32),
            'filter/cbf_violation': jnp.max(-residual[:, :data['mask'].shape[1]]*data['mask'], axis=1, initial=0.),
            'filter/primal_residual': primal,
            'filter/kkt_residual': jnp.max(errors[:, 1:], axis=1),
            'filter/initial_safe': jnp.all(data['constraints'].domain >= -1e-5, axis=1).astype(jnp.float32),
        }
    result = jax.tree_util.tree_map(jax.lax.stop_gradient, (action, diagnostics))
    return jax.tree_util.tree_map(lambda value: value[0], result) if single else result
