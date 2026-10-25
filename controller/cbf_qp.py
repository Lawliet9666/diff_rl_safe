"""CPU OSQP safety filters for robot CBF constraints."""
from typing import NamedTuple
from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import osqp
from scipy import sparse

from env.robots import SingleIntegratorRobot, HumanoidRobot
from env.robots.base import check_obstacle_shapes


def select_obstacles(position, obstacles, top_k):
    """Select nearest obstacles without changing scene indices or array shapes."""
    if top_k is None:
        return obstacles
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("controller.qp_obs_top_k must be null or a positive integer")
    distance = jnp.linalg.norm(obstacles.centers-position, axis=1)
    _, indices = jax.lax.top_k(-jnp.where(obstacles.mask > .5, distance, jnp.inf),
                              min(top_k, obstacles.mask.size))
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


def _tracking_qp_data(nominal, constraints, low, high, weights, output_matrix):
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
    MS = M*scale[None, :]
    P = MS.T @ (weights[:, None]*MS)
    return nominal, low, high, cbf_A, cbf_b, P, scale


def _solve_qp_data(nominal, A, b, P, scale, domain, solver):
    """Solve the scaled QP and check feasibility/KKT residuals."""
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
    q = -P @ (nominal/scale)
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
    if result.x is not None and result.y is not None:
        z, dual = np.asarray(result.x), np.asarray(result.y)
        slack = C @ z - lower
        kkt = np.array([max(0., float(np.max(-slack))),
                        np.max(np.abs(P@z + q + C.T@dual)),
                        max(float(np.max(np.abs(slack*dual))), float(np.max(dual)), 0.)])
    valid = bool(status == "solved" and np.isfinite(candidate).all()
                 and np.isfinite(residual).all() and np.all(residual >= -tolerance)
                 and np.isfinite(kkt).all() and np.all(kkt <= tolerance))
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


def solve_qp(nominal, constraints, low, high, weights, *,
             output_matrix, solver=SolverConfig()):
    """Minimize 0.5*||M@(u-nominal)||_W^2 with hard constraints (CPU OSQP)."""
    nominal, low, high, cbf_A, cbf_b, P, scale = _tracking_qp_data(
        nominal, constraints, low, high, weights, output_matrix)
    d = nominal.size
    A = np.concatenate([cbf_A, np.eye(d), -np.eye(d)])
    b = np.concatenate([cbf_b, -low, high])
    return _solve_qp_data(nominal, A, b, P, scale, constraints.domain, solver)


def validate_slack_weight(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("controller.slack_weight must be finite and nonnegative")


def solve_slack_qp(nominal, constraints, low, high, weights, *, soft_mask,
                   slack_weight, output_matrix, solver=SolverConfig()):
    """CPU QP with cost 0.5*||M@(u-nominal)||_W^2 + rho/2*||delta||^2.

    Only soft_mask rows receive slacks, in constraint order.
    residual excludes slack; valid checks the relaxed QP."""
    validate_slack_weight(slack_weight)
    if slack_weight == 0:
        raise ValueError("solve_slack_qp requires a positive slack_weight")
    nominal, low, high, cbf_A, cbf_b, tracking_P, control_scale = _tracking_qp_data(
        nominal, constraints, low, high, weights, output_matrix)
    n, d = cbf_A.shape
    mask = np.asarray(soft_mask)
    if mask.shape != (n,) or not np.isfinite(mask).all():
        raise ValueError("soft_mask must be a finite vector with one entry per CBF row")
    ids = np.flatnonzero(mask > .5)
    k = len(ids)
    E = np.zeros((n, k))
    E[ids, np.arange(k)] = 1.
    A = np.block([
        [cbf_A, E],
        [np.eye(d), np.zeros((d, k))],
        [-np.eye(d), np.zeros((d, k))],
        [np.zeros((k, d)), np.eye(k)],
    ])
    b = np.concatenate([cbf_b, -low, high, np.zeros(k)])
    # delta = z_delta/sqrt(rho) conditions the cost.
    scale = np.concatenate([control_scale, np.full(k, 1./math.sqrt(slack_weight))])
    P = np.zeros((d+k, d+k))
    P[:d, :d] = tracking_P
    P[d:, d:] = np.eye(k)
    augmented = _solve_qp_data(
        np.concatenate([nominal, np.zeros(k)]), A, b, P, scale,
        constraints.domain, solver)
    candidate = augmented.candidate[:d]
    raw = jnp.concatenate([
        jnp.asarray(cbf_A)@candidate+jnp.asarray(cbf_b),
        candidate-jnp.asarray(low), jnp.asarray(high)-candidate])
    result = augmented._replace(
        action=augmented.action[:d], candidate=candidate, residual=raw,
        correction_norm=float(np.linalg.norm(np.asarray(candidate)-nominal)))
    slack = jnp.zeros(n, dtype=candidate.dtype).at[ids].set(augmented.candidate[d:])
    return result, slack


@partial(jax.jit, static_argnames=("robot",))
def _control_output_matrix(robot, state):
    return robot.control_output_matrix(state)


@partial(jax.jit, static_argnames=("robot",))
def _controller_constraints(robot, state, obstacles, dt):
    constraints = robot.cbf_constraints(state, obstacles, dt)
    if isinstance(robot, SingleIntegratorRobot):
        # Match SF-PPO's 16-facet inner speed polygon; keep step() rescaling inactive.
        angles = jnp.arange(16)*2*jnp.pi/16
        normals = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        bound = robot.v_max*math.cos(math.pi/16)*(1.-1e-7)
        constraints = constraints._replace(
            A=jnp.concatenate([constraints.A, -normals]),
            b=jnp.concatenate([constraints.b, jnp.full(16, bound)]))
    return constraints


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


def _solve_controller_qp(robot, state, nominal, obstacles, dt, *, solver, slack_weight):
    if isinstance(robot, HumanoidRobot) and not bool(robot.barrier_domain_valid(state, obstacles)):
        raise ValueError("parabolic CBF requires strict separation from inflated obstacles")
    low, high = robot.input_bounds()
    constraints = _controller_constraints(robot, state, obstacles, dt)
    weights = jnp.asarray(robot.weights)
    options = dict(solver=solver,
                   output_matrix=_control_output_matrix(robot, state))
    if slack_weight == 0:
        result = solve_qp(nominal, constraints, low, high, weights, **options)
        return result, jnp.zeros_like(obstacles.mask)
    # Leading rows are obstacles; appended state/speed constraints stay hard.
    soft_mask = jnp.pad(obstacles.mask, (0, constraints.A.shape[0]-obstacles.mask.size))
    result, slack = solve_slack_qp(nominal, constraints, low, high, weights,
                                 soft_mask=soft_mask, slack_weight=slack_weight, **options)
    # An infeasibility certificate is not a slack solution for the held input.
    applied_slack = slack[:obstacles.mask.size] if result.valid else jnp.zeros_like(obstacles.mask)
    return result, applied_slack


def solve_cbf_qp(robot, state, nominal, obstacles, dt, *, solver=SolverConfig()):
    _validate_controller_inputs(robot, state, nominal, obstacles, dt)
    result, _ = _solve_controller_qp(robot, state, nominal, obstacles, dt,
                                     solver=solver, slack_weight=0.0)
    return result


def warmup_controller(robot, state, nominal, obstacles, dt, *, qp_obs_top_k=None):
    """Compile CBF/output computations without solving a QP."""
    _validate_controller_inputs(robot, state, nominal, obstacles, dt)
    obstacles = select_obstacles(state[:2], obstacles, qp_obs_top_k)
    if isinstance(robot, HumanoidRobot) and not bool(robot.barrier_domain_valid(state, obstacles)):
        raise ValueError("parabolic CBF requires strict separation from inflated obstacles")
    jax.block_until_ready((_controller_constraints(robot, state, obstacles, dt),
                           _control_output_matrix(robot, state)))


def generate_action(robot, state, nominal, obstacles, dt, *, solver=SolverConfig(),
                    slack_weight=0.0, qp_obs_top_k=None):
    """Non-differentiable CPU OSQP filter for a nominal physical input [control_dim].

    slack_weight=0 is hard; positive weights relax active obstacle rows only.
    Full-scene input and returned obstacle_mask retain scene ordering.
    Numerical errors raise; infeasibility returns qp.valid=False and a NaN action."""
    validate_slack_weight(slack_weight)
    _validate_controller_inputs(robot, state, nominal, obstacles, dt)
    obstacles = select_obstacles(state[:2], obstacles, qp_obs_top_k)
    result, slack = _solve_controller_qp(robot, state, nominal, obstacles, dt,
                                        solver=solver, slack_weight=slack_weight)
    return result.action, {"nominal": nominal, "qp": result, "slack": slack,
                           "obstacle_mask": obstacles.mask}
