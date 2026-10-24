"""CPU OSQP safety filtering over pure JAX robot constraints; no differentiation."""
from typing import NamedTuple
from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import osqp
from scipy import sparse

from env.robots import UnicycleRobot, QuadrotorRobot, HumanoidRobot
from env.robots.cbf import Obstacles


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


@dataclass(frozen=True)
class UnicycleConfig:
    goal_tolerance: float
    radius: float = 0.3
    margin: float = 0.05
    lookahead: float = 0.2
    alpha: float = 2.0
    v_max: float = UnicycleRobot.v_max
    omega_max: float = UnicycleRobot.omega_max
    weights: tuple = (1., 1.)
    position_gain: float = 1.0
    solver: SolverConfig = SolverConfig()

    def __post_init__(self):
        vals = (self.radius, self.lookahead, self.alpha, self.v_max, self.omega_max,
                self.position_gain, self.goal_tolerance, *self.weights)
        if (not all(math.isfinite(v) and v > 0 for v in vals)
                or not math.isfinite(self.margin) or self.margin < 0 or len(self.weights) != 2):
            raise ValueError("radii, lookahead, gains and limits must be positive")


@dataclass(frozen=True)
class QuadrotorConfig:
    mass: float = QuadrotorRobot.mass
    gravity: float = QuadrotorRobot.gravity
    radius: float = 0.3
    margin: float = 0.05
    gains: tuple = (1.5, 1.5, 1.5)
    omega_max: float = QuadrotorRobot.omega_max
    thrust_rate_max: float = QuadrotorRobot.thrust_rate_max
    thrust_min: float = 2.0
    thrust_max: float = 20.0
    thrust_gain: float = 4.0
    tilt_max: float = math.pi/4
    tilt_gain: float = 4.0
    altitude: float = QuadrotorRobot.altitude
    altitude_min: float = 0.8
    altitude_max: float = 1.6
    weights: tuple = (1., 1., 1., 1e-6)
    position_gain: float = 1.0
    velocity_gain: float = 1.8
    attitude_gain: float = 4.0
    thrust_tracking_gain: float = 4.0
    acceleration_limits: tuple = (2., 2., 2.)
    solver: SolverConfig = SolverConfig()

    def __post_init__(self):
        positives = (self.mass, self.gravity, self.radius, *self.gains,
                     self.omega_max, self.thrust_rate_max, self.thrust_gain,
                     self.tilt_gain, self.thrust_min, *self.weights, self.position_gain,
                     self.velocity_gain, self.attitude_gain, self.thrust_tracking_gain,
                     *self.acceleration_limits)
        if (not all(math.isfinite(v) and v > 0 for v in positives)
                or not math.isfinite(self.margin) or self.margin < 0 or len(self.gains) != 3
                or len(self.weights) != 4 or len(self.acceleration_limits) != 3
                or not 0 < self.tilt_max < math.pi/2
                or not self.altitude_min < self.altitude < self.altitude_max
                or not self.thrust_min < self.mass*self.gravity < self.thrust_max):
            raise ValueError("invalid quadrotor gains, physical limits or hover configuration")


@dataclass(frozen=True)
class HumanoidConfig:
    goal_tolerance: float
    dt: float
    radius: float = 0.3
    margin: float = 0.05
    barrier: str = "distance"
    gains: tuple = (1.5, 1.5)
    alpha: float = 2.0
    parabolic_gain: float = 1.0
    clearance_gain: float = 1.0
    speed_epsilon: float = 0.05
    weights: tuple = (1., 1., 1.)
    position_gain: float = 1.0
    velocity_gain: float = 2.0
    heading_gain: float = 2.0
    solver: SolverConfig = SolverConfig()

    def __post_init__(self):
        values = (self.goal_tolerance, self.dt, self.radius, *self.gains,
                  self.alpha, self.parabolic_gain, self.clearance_gain,
                  self.speed_epsilon, *self.weights, self.position_gain,
                  self.velocity_gain, self.heading_gain)
        if (self.barrier not in {"distance", "parabolic"}
                or len(self.gains) != 2 or len(self.weights) != 3
                or not all(math.isfinite(v) and v > 0 for v in values)
                or not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError("invalid humanoid barrier, gains or controller limits")


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


@partial(jax.jit, static_argnames=("robot", "control_config"))
def _nominal_control(robot, state, goal, control_config):
    return robot.nominal_control(state, goal, control_config)


@partial(jax.jit, static_argnames=("robot", "cbf_config"))
def _cbf_constraints(robot, state, obstacles, cbf_config):
    return robot.cbf_constraints(state, obstacles, cbf_config)


@partial(jax.jit, static_argnames=("robot", "control_config"))
def _control_output_matrix(robot, state, control_config):
    return robot.control_output_matrix(state, control_config)


def control_residual(robot, state, action, obstacles, control_config):
    """Evaluate the actual applied input, including a requested fallback input."""
    constraints = _cbf_constraints(robot, state, obstacles, control_config)
    low, high = robot.input_bounds()
    return jnp.concatenate([constraints.A@action+constraints.b, action-low, high-action])


def solve_cbf_qp(robot, state, nominal, obstacles, cbf_config):
    """Apply one shared CPU QP to the selected robot's constraints and bounds."""
    if isinstance(robot, HumanoidRobot) and not bool(
            robot.barrier_domain_valid(state, obstacles, cbf_config)):
        # Do not extend the parabolic sqrt through collision or execute an old
        # acceleration. Return an explicit failed result before calling OSQP.
        action = jnp.full(robot.control_dim, jnp.nan)
        residual = jnp.full(obstacles.mask.size+4+2*robot.control_dim, jnp.nan)
        return ControlResult(action, action, residual, jnp.full(3, jnp.inf),
                             False, False, float("nan"), "invalid barrier domain", 0)
    low, high = robot.input_bounds()
    return solve_qp(nominal, _cbf_constraints(robot, state, obstacles, cbf_config),
                        low, high, jnp.asarray(cbf_config.weights),
                        solver=cbf_config.solver,
                        output_matrix=_control_output_matrix(robot, state, cbf_config))


def generate_action(robot, state, goal, obstacles, control_config):
    """Return a physical action and diagnostics for any supported robot.

    Only nominal/CBF calculations are JIT-compiled. The CPU OSQP solve is neither
    JIT-compatible nor differentiable. Invalid QP results keep their NaN action;
    callers must inspect qp.valid and qp.initial_safe before stepping.
    """
    nominal = _nominal_control(robot, state, goal, control_config)
    result = solve_cbf_qp(robot, state, nominal, obstacles, control_config)
    return result.action, {"nominal": nominal, "qp": result,
                           "slack": jnp.zeros_like(obstacles.mask)}
