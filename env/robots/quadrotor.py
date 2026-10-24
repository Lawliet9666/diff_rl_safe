"""Attitude-aware quadrotor: [p(3), v(3), row-major R(9), thrust(1)]."""
from dataclasses import KW_ONLY, dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, Constraints, check_obstacle_shapes,
                   masked_rows, join_constraints, _hocbf)


def skew(w):
    x, y, z = w
    return jnp.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def unpack(state):
    if state.shape != (16,):
        raise ValueError("quadrotor state must be [p(3),v(3),R(9),T], shape (16,)")
    return state[:3], state[3:6], state[6:15].reshape(3, 3), state[15]


def _rotation_increment(omega, dt):
    w = omega*dt
    theta = jnp.sqrt(w@w+1e-24)
    K = skew(w)
    return jnp.eye(3)+jnp.sinc(theta/jnp.pi)*K+0.5*jnp.sinc(theta/(2*jnp.pi))**2*(K@K)


@dataclass(frozen=True)
class QuadrotorRobot(Robot):
    mass: float = 1.0
    gravity: float = 9.81
    altitude: float = 1.2
    omega_max: float = 2.0
    thrust_rate_max: float = 20.0
    state_dim: ClassVar[int] = 16
    control_dim: ClassVar[int] = 4

    # Optional settings; ordinary environment construction needs only the physical fields above.
    _: KW_ONLY

    # Collision geometry and clearance.
    radius: float = 0.3
    margin: float = 0.05

    # State limits.
    thrust_min: float = 2.0
    thrust_max: float = 20.0
    tilt_max: float = math.pi / 4
    altitude_min: float = 0.8
    altitude_max: float = 1.6

    # Nominal tracking.
    position_gain: float = 1.0
    velocity_gain: float = 1.8
    attitude_gain: float = 4.0
    thrust_tracking_gain: float = 4.0
    acceleration_limits: tuple = (2.0, 2.0, 2.0)

    # CBF decay.
    gains: tuple = (1.5, 1.5, 1.5)
    thrust_gain: float = 4.0
    tilt_gain: float = 4.0

    # QP tracking weights.
    weights: tuple = (1.0, 1.0, 1.0, 1e-06)

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

    def hover_state(self, position=None):
        p = jnp.array([0., 0., self.altitude]) if position is None else jnp.asarray(position)
        return jnp.concatenate([p, jnp.zeros(3), jnp.eye(3).reshape(-1),
                                jnp.array([self.mass*self.gravity])])

    def acceleration(self, state):
        _, _, R, T = unpack(state)
        return jnp.array([0., 0., -self.gravity])+T/self.mass*R[:, 2]

    def jerk_map(self, state):
        _, _, R, T = unpack(state)
        return jnp.column_stack([-T/self.mass*R@skew(jnp.array([0., 0., 1.])),
                                 R[:, 2]/self.mass])

    def control_output_matrix(self, state):
        """Track world jerk; the last row resolves the unobservable yaw direction."""
        return jnp.vstack([self.jerk_map(state), jnp.array([0., 0., 1., 0.])])

    def dynamics(self, state, action):
        _, v, R, _ = unpack(state)
        return jnp.concatenate([v, self.acceleration(state),
                                (R@skew(action[:3])).reshape(-1), action[3:4]])

    def step_with_info(self, state, action, dt):
        """Midpoint translation, exact held-rate rotation; no additional clipping."""
        p, v, R, T = unpack(state)
        Rmid = R@_rotation_increment(action[:3], dt/2)
        amid = jnp.array([0., 0., -self.gravity])+(T+action[3]*dt/2)/self.mass*Rmid[:, 2]
        next_state = jnp.concatenate([p+v*dt+0.5*amid*dt**2, v+amid*dt,
            (R@_rotation_increment(action[:3], dt)).reshape(-1),
            jnp.array([T+action[3]*dt])])
        return RobotStep(next_state, next_state[3:6], action)

    def input_bounds(self):
        limits = jnp.array([self.omega_max]*3+[self.thrust_rate_max])
        return -limits, limits

    def navigation_state(self, state):
        R = state[6:15].reshape(3, 3)
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return jnp.array([state[0], state[1], yaw])

    def barrier(self, state, obstacles):
        """Return h, h_dot, h_ddot, dh_ddot_dx_g and h_dddot_drift.

        dh_ddot_dx_g = (d h_ddot / d state) @ g(state).
        h_dddot = dh_ddot_dx_g@u+h_dddot_drift.
        """
        p, v, _, _ = unpack(state)
        a, J = self.acceleration(state), self.jerk_map(state)
        rel, w = p[None, :2]-obstacles.centers, v[None, :2]-obstacles.velocities
        h = jnp.sum(rel*rel, axis=-1)-(self.radius+obstacles.radii+self.margin)**2
        h_dot = 2*jnp.sum(rel*w, axis=1)
        h_ddot = 2*jnp.sum(w*w, axis=1)+2*rel@a[:2]
        dh_ddot_dx_g = 2*rel@J[:2]
        h_dddot_drift = 6*w@a[:2]
        return h, h_dot, h_ddot, dh_ddot_dx_g, h_dddot_drift

    def altitude_barrier(self, state):
        """Floor/ceiling h, h_dot, h_ddot, dh_ddot_dx_g and h_dddot_drift.

        h_dddot = dh_ddot_dx_g@u+h_dddot_drift; decay gains are external.
        """
        p, v, _, _ = unpack(state)
        a, J = self.acceleration(state), self.jerk_map(state)
        normals = jnp.array([[0., 0., 1.], [0., 0., -1.]])
        offset = jnp.array([-self.altitude_min, self.altitude_max])
        h = normals@p+offset
        h_dot = normals@v
        h_ddot = normals@a
        dh_ddot_dx_g = normals@J
        h_dddot_drift = jnp.zeros(2)
        return h, h_dot, h_ddot, dh_ddot_dx_g, h_dddot_drift

    def state_barriers(self, state):
        """Return tilt/thrust h and dh_dx_g, with h_dot = dh_dx_g@u."""
        _, _, R, T = unpack(state)
        h = jnp.array([R[2, 2]-math.cos(self.tilt_max),
                       T-self.thrust_min, self.thrust_max-T])
        tilt_dh_dx_g = jnp.concatenate([(-R@skew(jnp.array([0., 0., 1.])))[2], jnp.zeros(1)])
        dh_dx_g = jnp.stack([tilt_dh_dx_g, jnp.array([0., 0., 0., 1.]),
                             jnp.array([0., 0., 0., -1.])])
        return h, dh_dx_g

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt):
        """Assemble obstacle/altitude HOCBFs and tilt/thrust state constraints."""
        check_obstacle_shapes(obstacles)
        obs = _hocbf(*self.barrier(state, obstacles), gains=self.gains)
        obs = masked_rows(obs.A, obs.b, obs.domain.reshape((-1, 3)), obstacles.mask)
        altitude = _hocbf(*self.altitude_barrier(state), gains=self.gains)
        h, dh_dx_g = self.state_barriers(state)
        b = jnp.array([self.tilt_gain*h[0], self.thrust_gain*h[1],
                       self.thrust_gain*h[2]])
        return join_constraints(obs, altitude, Constraints(dh_dx_g, b, h))

    def nominal_control(self, state, goal):
        """Goal-tracking reference; CBF-QP filtering remains external."""
        p, v, R, T = unpack(state)
        goal = jnp.concatenate([goal, jnp.array([self.altitude])])
        desired_acc = self.position_gain*(goal-p)-self.velocity_gain*v
        limits = jnp.asarray(self.acceleration_limits)
        desired_acc = jnp.clip(desired_acc, -limits, limits)
        force = self.mass*(desired_acc+jnp.array([0., 0., self.gravity]))
        bdes = force/jnp.maximum(jnp.linalg.norm(force), 1e-9)
        # Align body thrust with desired force. Yaw is unregulated in this 2D task.
        omega = self.attitude_gain*jnp.cross(jnp.array([0., 0., 1.]), R.T@bdes)
        nu = self.thrust_tracking_gain*(jnp.linalg.norm(force)-T)
        return jnp.concatenate([omega, jnp.array([nu])])
