"""Model-specific JAX CBF constraints and shared CPU OSQP safety filtering."""
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
from env.robots.quadrotor import skew, unpack


class Obstacles(NamedTuple):
    """Single scene: centers/velocities [N,2], radii/mask [N].

    Velocities are world-frame and held constant in the barrier prediction.
    Masked slots may contain arbitrary finite values; they add no constraint.
    """
    centers: jax.Array
    velocities: jax.Array
    radii: jax.Array
    mask: jax.Array


class Constraints(NamedTuple):
    """A u + b >= 0; domain contains h and recursive HOCBF initial values."""
    A: jax.Array
    b: jax.Array
    domain: jax.Array


def check_obstacle_shapes(obstacles):
    n = obstacles.centers.shape[0]
    if (obstacles.centers.shape != (n, 2)
            or obstacles.velocities.shape != (n, 2)
            or obstacles.radii.shape != (n,) or obstacles.mask.shape != (n,)):
        raise ValueError("expected centers/velocities [N,2] and radii/mask [N]")


def masked_rows(A, b, domain, mask):
    active = mask > 0.5
    # A strictly feasible 0*u+1>=0 row avoids a degenerate masked equality.
    return Constraints(jnp.where(active[:, None], A, 0.0),
                       jnp.where(active, b, 1.0),
                       jnp.where(active[:, None], domain, 1.0).reshape(-1))


def join_constraints(*blocks):
    return Constraints(*(jnp.concatenate([b[i] for b in blocks], axis=0)
                         for i in range(3)))


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


def unicycle_constraints(robot, state, obstacles, cbf_config):
    """First-order obstacle CBFs for the unicycle lookahead point."""
    check_obstacle_shapes(obstacles)
    theta = state[2]
    heading = jnp.array([jnp.cos(theta), jnp.sin(theta)])
    q = state[:2]+cbf_config.lookahead*heading
    B = robot.control_output_matrix(state, cbf_config)
    rel = q-obstacles.centers
    # Inflating by lookahead also separates the actual robot disk.
    distance = cbf_config.radius+obstacles.radii+cbf_config.margin+cbf_config.lookahead
    h = jnp.sum(rel*rel, axis=-1)-distance**2
    A = 2*rel@B
    b = cbf_config.alpha*h-2*jnp.sum(rel*obstacles.velocities, axis=1)
    return masked_rows(A, b, h[:, None], obstacles.mask)


def _hocbf(h, h1, h2, A3, b3, gains):
    k1, k2, k3 = gains
    psi1 = h1+k1*h
    psi2 = h2+(k1+k2)*h1+k1*k2*h
    b = b3+(k1+k2+k3)*h2+(k1*k2+k1*k3+k2*k3)*h1+k1*k2*k3*h
    domain = jnp.stack([h, psi1, psi2], axis=-1).reshape(-1)
    return Constraints(A3, b, domain)


def quadrotor_constraints(robot, state, obstacles, cbf_config):
    """Obstacle/altitude HOCBFs plus tilt and thrust state constraints."""
    check_obstacle_shapes(obstacles)
    p, v, R, T = unpack(state)
    a, J = robot.acceleration(state), robot.jerk_map(state)
    rel, w = p[None, :2]-obstacles.centers, v[None, :2]-obstacles.velocities
    h = jnp.sum(rel*rel, axis=-1)-(cbf_config.radius+obstacles.radii+cbf_config.margin)**2
    h1 = 2*jnp.sum(rel*w, axis=1)
    h2 = 2*jnp.sum(w*w, axis=1)+2*rel@a[:2]
    obs = _hocbf(h, h1, h2, 2*rel@J[:2], 6*w@a[:2], cbf_config.gains)
    obs = masked_rows(obs.A, obs.b, obs.domain.reshape((-1, 3)), obstacles.mask)
    normals = jnp.array([[0., 0., 1.], [0., 0., -1.]])
    offset = jnp.array([-cbf_config.altitude_min, cbf_config.altitude_max])
    altitude = _hocbf(normals@p+offset, normals@v, normals@a,
                       normals@J, jnp.zeros(2), cbf_config.gains)
    htilt = R[2, 2]-math.cos(cbf_config.tilt_max)
    tilt_A = jnp.concatenate([(-R@skew(jnp.array([0., 0., 1.])))[2], jnp.zeros(1)])
    limits = Constraints(jnp.stack([tilt_A, jnp.array([0., 0., 0., 1.]),
                                    jnp.array([0., 0., 0., -1.])]),
                         jnp.array([cbf_config.tilt_gain*htilt,
                                    cbf_config.thrust_gain*(T-cbf_config.thrust_min),
                                    cbf_config.thrust_gain*(cbf_config.thrust_max-T)]),
                         jnp.array([htilt, T-cbf_config.thrust_min, cbf_config.thrust_max-T]))
    return join_constraints(obs, altitude, limits)


def _parabolic_value(z, w, radius, cfg):
    distance = jnp.linalg.norm(z)
    e = z / distance
    tangent = jnp.array([-e[1], e[0]])
    clearance = jnp.sqrt(z @ z - radius**2)
    lam = cfg.parabolic_gain * clearance / jnp.sqrt(w @ w + cfg.speed_epsilon**2)
    return e @ w + lam * (tangent @ w)**2 + cfg.clearance_gain * clearance


def _humanoid_barrier_domain_valid(state, obstacles, cfg):
    """The parabolic formula is defined strictly outside inflated obstacles."""
    check_obstacle_shapes(obstacles)
    if cfg.barrier == "distance":
        return jnp.array(True)
    radius = cfg.radius + obstacles.radii + cfg.margin
    q = jnp.sum((state[:2]-obstacles.centers)**2, axis=1) - radius**2
    return jnp.all((obstacles.mask <= .5) | (q > 0))


def humanoid_constraints(robot, state, obstacles, cbf_config):
    """Distance or parabolic obstacle CBFs and next-step body velocity bounds."""
    check_obstacle_shapes(obstacles)
    cfg = cbf_config
    z = state[:2]-obstacles.centers
    w = robot.world_velocity(state)-obstacles.velocities
    radius = cfg.radius + obstacles.radii + cfg.margin
    B = robot.acceleration_map(state)
    if cfg.barrier == "distance":
        h = jnp.sum(z*z, axis=1)-radius**2
        h1 = 2*jnp.sum(z*w, axis=1)
        k1, k2 = cfg.gains
        A = 2*z @ B
        b = 2*jnp.sum(w*w, axis=1)+(k1+k2)*h1+k1*k2*h
        domain = jnp.stack([h, h1+k1*h], axis=1)
    elif cfg.barrier == "parabolic":
        # Inactive padding may lie inside a disk; give it an irrelevant,
        # valid evaluation point before differentiating, then mask its rows.
        dummy = jnp.stack([radius+1., jnp.zeros_like(radius)], axis=1)
        z = jnp.where((obstacles.mask > .5)[:, None], z, dummy)
        w = jnp.where((obstacles.mask > .5)[:, None], w, 0.)
        value = lambda z, w, r: _parabolic_value(z, w, r, cfg)
        h = jax.vmap(value)(z, w, radius)
        dz, dw = jax.vmap(jax.grad(value, argnums=(0, 1)))(z, w, radius)
        A = dw @ B
        b = jnp.sum(dz*w, axis=1)+cfg.alpha*h
        domain = jnp.stack([h, jnp.sum(z*z, axis=1)-radius**2], axis=1)
    else:
        raise ValueError("humanoid barrier must be distance or parabolic")
    obstacle_rows = masked_rows(A, b, domain, obstacles.mask)
    # The ideal model's body velocities are its command-integrator states.
    # These rows constrain the next command; the integrator never clips it.
    margin = robot.velocity_margin(state)
    velocity_rows = Constraints(
        cfg.dt*jnp.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]),
        margin, margin)
    return join_constraints(obstacle_rows, velocity_rows)


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


@partial(jax.jit, static_argnames=("robot", "cbf_config"))
def _cbf_constraints(robot, state, obstacles, cbf_config):
    if isinstance(robot, UnicycleRobot):
        return unicycle_constraints(robot, state, obstacles, cbf_config)
    if isinstance(robot, QuadrotorRobot):
        return quadrotor_constraints(robot, state, obstacles, cbf_config)
    if isinstance(robot, HumanoidRobot):
        return humanoid_constraints(robot, state, obstacles, cbf_config)
    raise NotImplementedError(
        f"CBF constraints are not implemented for {type(robot).__name__}")


@partial(jax.jit, static_argnames=("robot", "control_config"))
def _control_output_matrix(robot, state, control_config):
    return robot.control_output_matrix(state, control_config)


def control_residual(robot, state, action, obstacles, control_config):
    """Evaluate the actual applied input, including a requested fallback input."""
    constraints = _cbf_constraints(robot, state, obstacles, control_config)
    low, high = robot.input_bounds()
    return jnp.concatenate([constraints.A@action+constraints.b, action-low, high-action])


def _solve_controller_qp(robot, state, nominal, obstacles, cbf_config, *, slack_weight):
    """Prepare shared QP inputs, then select hard or relaxed obstacle constraints."""
    if isinstance(robot, HumanoidRobot) and not bool(
            _humanoid_barrier_domain_valid(state, obstacles, cbf_config)):
        # Do not extend the parabolic sqrt through collision or execute an old
        # acceleration. Return an explicit failed result before calling OSQP.
        action = jnp.full(robot.control_dim, jnp.nan)
        residual = jnp.full(obstacles.mask.size+4+2*robot.control_dim, jnp.nan)
        result = ControlResult(action, action, residual, jnp.full(3, jnp.inf),
                               False, False, float("nan"), "invalid barrier domain", 0)
        return result, jnp.zeros_like(obstacles.mask)
    low, high = robot.input_bounds()
    constraints = _cbf_constraints(robot, state, obstacles, cbf_config)
    weights = jnp.asarray(cbf_config.weights)
    options = dict(solver=cbf_config.solver,
                   output_matrix=_control_output_matrix(robot, state, cbf_config))
    if slack_weight == 0:
        result = solve_qp(nominal, constraints, low, high, weights, **options)
        return result, jnp.zeros_like(obstacles.mask)
    return solve_slack_qp(nominal, constraints, low, high, weights,
                          soft_mask=obstacles.mask, slack_weight=slack_weight, **options)


def solve_cbf_qp(robot, state, nominal, obstacles, cbf_config):
    """Apply a hard CBF-QP to an externally supplied nominal input."""
    result, _ = _solve_controller_qp(robot, state, nominal, obstacles, cbf_config,
                                     slack_weight=0.0)
    return result


def generate_action(robot, state, nominal, obstacles, control_config, *, slack_weight=0.0):
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
    result, slack = _solve_controller_qp(robot, state, nominal, obstacles, control_config,
                                        slack_weight=slack_weight)
    return result.action, {"nominal": nominal, "qp": result, "slack": slack}
