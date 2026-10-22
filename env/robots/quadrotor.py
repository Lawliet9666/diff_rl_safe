"""Attitude-aware quadrotor: [p(3), v(3), row-major R(9), thrust(1)]."""
from dataclasses import dataclass
import math
from typing import ClassVar

import jax.numpy as jnp

from .base import Robot, RobotStep
from .cbf import Constraints, check_obstacle_shapes, masked_rows, join_constraints


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


def _hocbf(h, h1, h2, A3, b3, gains):
    k1, k2, k3 = gains
    psi1 = h1+k1*h
    psi2 = h2+(k1+k2)*h1+k1*k2*h
    b = b3+(k1+k2+k3)*h2+(k1*k2+k1*k3+k2*k3)*h1+k1*k2*k3*h
    domain = jnp.stack([h, psi1, psi2], axis=-1).reshape(-1)
    return Constraints(A3, b, domain)


@dataclass(frozen=True)
class QuadrotorRobot(Robot):
    mass: float = 1.0
    gravity: float = 9.81
    altitude: float = 1.2
    omega_max: float = 2.0
    thrust_rate_max: float = 20.0
    state_dim: ClassVar[int] = 16
    control_dim: ClassVar[int] = 4

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

    def control_output_matrix(self, state, control_config):
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

    def barrier(self, state, centers, radii, cfg):
        rel = state[:2]-centers
        return jnp.sum(rel*rel, axis=-1)-(cfg.radius+radii+cfg.margin)**2

    def cbf_constraints(self, state, obstacles, cbf_config):
        check_obstacle_shapes(obstacles)
        p, v, R, T = unpack(state)
        a, J = self.acceleration(state), self.jerk_map(state)
        rel, w = p[None, :2]-obstacles.centers, v[None, :2]-obstacles.velocities
        h = self.barrier(state, obstacles.centers, obstacles.radii, cbf_config)
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

    def nominal_control(self, state, goal, control_config):
        """Goal-tracking reference; CBF-QP filtering remains external."""
        p, v, R, T = unpack(state)
        goal = jnp.concatenate([goal, jnp.array([self.altitude])])
        desired_acc = control_config.position_gain*(goal-p)-control_config.velocity_gain*v
        limits = jnp.asarray(control_config.acceleration_limits)
        desired_acc = jnp.clip(desired_acc, -limits, limits)
        force = self.mass*(desired_acc+jnp.array([0., 0., self.gravity]))
        bdes = force/jnp.maximum(jnp.linalg.norm(force), 1e-9)
        # Align body thrust with desired force. Yaw is unregulated in this 2D task.
        omega = control_config.attitude_gain*jnp.cross(jnp.array([0., 0., 1.]), R.T@bdes)
        nu = control_config.thrust_tracking_gain*(jnp.linalg.norm(force)-T)
        return jnp.concatenate([omega, jnp.array([nu])])
