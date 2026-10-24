"""Shared CPU OSQP safety filtering over robot-provided CBF constraints."""
from typing import NamedTuple
from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import osqp
from scipy import sparse

from env.robots import UnicycleRobot, HumanoidRobot
from env.robots.base import check_obstacle_shapes


def select_obstacles(position, obstacles, top_k):
    """Restrict QP visibility while preserving original scene indices and shapes."""
    if top_k is None:
        return obstacles
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
    feasibility_tolerance: float = 1e-6
    polishing: bool = False
    verbose: bool = False

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in
                    [self.eps_abs, self.eps_rel, self.feasibility_tolerance])
                or not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool)
                or self.max_iter <= 0
                or not isinstance(self.polishing, bool) or not isinstance(self.verbose, bool)):
            raise ValueError("invalid OSQP tolerances, iteration limit or logging settings")


class ControlResult(NamedTuple):
    action: jax.Array          # NaN when the candidate is invalid; never executed.
    candidate: jax.Array
    residual: jax.Array        # Original CBF/input residuals, without relaxation.
    kkt_residual: jax.Array    # Scaled primal, dual and complementarity residuals.
    valid: bool               # Solved QP satisfies its own constraints and KKT checks.
    initial_safe: bool         # All h/psi initial conditions hold.
    correction_norm: float
    solver_status: str
    iterations: int


def _tracking_qp_data(nominal, constraints, low, high, weights, output_matrix):
    """Validate physical data and build the scaled nominal-tracking objective."""
    if not jax.config.x64_enabled:
        raise ValueError("CBF-QP requires JAX_ENABLE_X64=true before importing JAX")
    nominal, low, high, weights = [np.asarray(x, dtype=np.float64)
                                   for x in (nominal, low, high, weights)]
    if nominal.ndim != 1:
        raise ValueError("nominal must be a vector")
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
    if (not all(np.isfinite(x).all() for x in (nominal, low, high, weights, M, cbf_A, cbf_b))
            or np.any(weights <= 0) or np.any(low > high)):
        raise ValueError("QP data must be finite, with positive weights and ordered bounds")
    MS = M*scale[None, :]
    P = MS.T @ (weights[:, None]*MS)
    return nominal, low, high, cbf_A, cbf_b, P, scale


def _solve_qp_data(nominal, A, b, P, scale, domain, solver):
    """Solve and independently check a scaled QP; all physical rows use A z+b>=0."""
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
    problem.setup(P=sparse.csc_matrix(np.triu(P)), q=q,
                  A=sparse.csc_matrix(C), l=lower, u=np.full(len(b), np.inf),
                  eps_abs=solver.eps_abs, eps_rel=solver.eps_rel,
                  max_iter=solver.max_iter, polishing=solver.polishing,
                  verbose=solver.verbose)
    result = problem.solve(raise_error=False)
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
    status = str(result.info.status)
    valid = bool(status == "solved" and np.isfinite(candidate).all()
                 and np.all(residual >= -tolerance)
                 and np.isfinite(kkt).all() and np.all(kkt <= tolerance))
    domain = np.asarray(domain)
    initial_safe = bool(np.isfinite(domain).all() and np.all(domain >= -tolerance))
    return ControlResult(jnp.asarray(candidate if valid else np.full(d, np.nan)),
                        jnp.asarray(candidate), jnp.asarray(residual), jnp.asarray(kkt),
                        valid, initial_safe, float(np.linalg.norm(candidate-nominal)),
                        status, int(result.info.iter))


def solve_qp(nominal, constraints, low, high, weights, *,
             output_matrix, solver=SolverConfig()):
    """Hard CPU QP with the original physical input limits and tracking cost.

    Neither JIT-compatible nor differentiable. With output_matrix=M, minimize
    0.5*||M@(u-nominal)||_W^2 using solver.feasibility_tolerance for validation.
    """
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
    """Minimize tracking cost + rho/2*||delta||^2 with A*u+b+E*delta>=0.

    Only rows selected by soft_mask receive nonnegative, unbounded slacks.
    Returned slack entries retain the original constraint-row ordering.
    valid checks the relaxed QP; residual still reports the original conditions.
    """
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
    # delta = z_delta/sqrt(rho): condition the cost without changing its units.
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


def control_residual(robot, state, action, obstacles, dt):
    """Evaluate the actual applied input, including a requested fallback input."""
    constraints = robot.cbf_constraints(state, obstacles, dt)
    low, high = robot.input_bounds()
    return jnp.concatenate([constraints.A@action+constraints.b, action-low, high-action])


def _solve_controller_qp(robot, state, nominal, obstacles, dt, *, solver, slack_weight):
    """Prepare shared QP inputs, then select hard or relaxed obstacle constraints."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(robot, HumanoidRobot):
        check_obstacle_shapes(obstacles)
        if not bool(robot.barrier_domain_valid(state, obstacles)):
            # Do not extend the parabolic sqrt through collision or execute an old
            # acceleration. Return an explicit failed result before calling OSQP.
            action = jnp.full(robot.control_dim, jnp.nan)
            residual = jnp.full(obstacles.mask.size+4+2*robot.control_dim, jnp.nan)
            result = ControlResult(action, action, residual, jnp.full(3, jnp.inf),
                                   False, False, float("nan"), "invalid barrier domain", 0)
            return result, jnp.zeros_like(obstacles.mask)
    low, high = robot.input_bounds()
    constraints = robot.cbf_constraints(state, obstacles, dt)
    weights = jnp.asarray(robot.weights)
    options = dict(solver=solver,
                   output_matrix=_control_output_matrix(robot, state))
    if slack_weight == 0:
        result = solve_qp(nominal, constraints, low, high, weights, **options)
        return result, jnp.zeros_like(obstacles.mask)
    return solve_slack_qp(nominal, constraints, low, high, weights,
                          soft_mask=obstacles.mask, slack_weight=slack_weight, **options)


def solve_cbf_qp(robot, state, nominal, obstacles, dt, *, solver=SolverConfig()):
    """Apply a hard CBF-QP to an externally supplied nominal input."""
    result, _ = _solve_controller_qp(robot, state, nominal, obstacles, dt,
                                     solver=solver, slack_weight=0.0)
    return result


def generate_action(robot, state, nominal, obstacles, dt, *, solver=SolverConfig(), slack_weight=0.0):
    """Filter a supplied nominal physical input; zero selects the hard QP.

    nominal has robot.control_dim entries in the robot input units.
    Slack currently relaxes only unicycle obstacle constraints. CBF calculations
    are JIT-compiled; the CPU OSQP solve is not differentiable.
    Invalid results keep their NaN action; callers must inspect qp.valid and
    qp.initial_safe before stepping.
    """
    validate_slack_weight(slack_weight)
    if slack_weight > 0 and not isinstance(robot, UnicycleRobot):
        raise ValueError("positive slack_weight currently supports unicycle only")
    result, slack = _solve_controller_qp(robot, state, nominal, obstacles, dt,
                                        solver=solver, slack_weight=slack_weight)
    return result.action, {"nominal": nominal, "qp": result, "slack": slack}
