"""Attitude-aware quadrotor: [p(3), v(3), row-major R(9), thrust(1)]."""
from dataclasses import dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, Constraints, check_obstacle_shapes,
                   masked_rows, join_constraints, _control_parameters, _hocbf)


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
    mass: float
    gravity: float
    altitude: float
    omega_max: float
    thrust_rate_max: float
    state_dim: ClassVar[int] = 16
    control_dim: ClassVar[int] = 4

    radius: float
    margin: float

    thrust_min: float
    thrust_max: float
    tilt_max: float
    altitude_min: float
    altitude_max: float

    position_gain: float
    velocity_gain: float
    attitude_gain: float
    thrust_tracking_gain: float
    acceleration_limits: tuple

    # Obstacle/altitude, thrust and tilt CBF rates.
    kappa: tuple
    kappa_thrust: float
    kappa_tilt: float

    weights: tuple

    @classmethod
    def from_config(cls, config):
        r = config.robot
        parameters = _control_parameters(
            config.controller.quadrotor,
            ("position_gain", "velocity_gain", "attitude_gain", "thrust_tracking_gain",
             "acceleration_limits", "kappa", "kappa_thrust", "kappa_tilt", "weights"),
            "controller.quadrotor")
        parameters.update(mass=r.mass, gravity=r.gravity, altitude=r.altitude,
                          omega_max=r.omega_max, thrust_rate_max=r.thrust_rate_max,
                          radius=r.radius, margin=float(config.env.safety_margin),
                          thrust_min=r.thrust_min, thrust_max=r.thrust_max,
                          tilt_max=r.tilt_max, altitude_min=r.altitude_min,
                          altitude_max=r.altitude_max)
        for name in ("kappa", "weights", "acceleration_limits"):
            parameters[name] = tuple(parameters[name])
        return cls(**parameters)

    def __post_init__(self):
        positives = (self.mass, self.gravity, self.radius, *self.kappa,
                     self.omega_max, self.thrust_rate_max, self.kappa_thrust,
                     self.kappa_tilt, self.thrust_min, *self.weights, self.position_gain,
                     self.velocity_gain, self.attitude_gain, self.thrust_tracking_gain,
                     *self.acceleration_limits)
        if (not all(math.isfinite(v) and v > 0 for v in positives)
                or not math.isfinite(self.margin) or self.margin < 0 or len(self.kappa) != 3
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
        """Track world jerk; the last row resolves body yaw."""
        return jnp.vstack([self.jerk_map(state), jnp.array([0., 0., 1., 0.])])

    def dynamics(self, state, action):
        _, v, R, _ = unpack(state)
        return jnp.concatenate([v, self.acceleration(state),
                                (R@skew(action[:3])).reshape(-1), action[3:4]])

    def step(self, state, action, dt):
        """Midpoint translation and exact held-rate rotation; no clipping."""
        p, v, R, T = unpack(state)
        Rmid = R@_rotation_increment(action[:3], dt/2)
        amid = jnp.array([0., 0., -self.gravity])+(T+action[3]*dt/2)/self.mass*Rmid[:, 2]
        next_state = jnp.concatenate([p+v*dt+0.5*amid*dt**2, v+amid*dt,
            (R@_rotation_increment(action[:3], dt)).reshape(-1),
            jnp.array([T+action[3]*dt])])
        return RobotStep(next_state, next_state[3:6], action)

    def initial_state(self, pose):
        state = self.hover_state(jnp.array([pose[0], pose[1], self.altitude]))
        yaw = pose[2].astype(state.dtype)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        R = jnp.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        return state.at[6:15].set(R.reshape(-1))

    def check_state_limits(self, state, tolerance=1e-6):
        limits = jnp.array([state[2]-self.altitude_min, self.altitude_max-state[2],
                            state[14]-math.cos(self.tilt_max),
                            state[15]-self.thrust_min, self.thrust_max-state[15]])
        return jnp.all(jnp.isfinite(state)) & jnp.all(limits >= -tolerance)

    def input_bounds(self):
        limits = jnp.array([self.omega_max]*3+[self.thrust_rate_max])
        return -limits, limits

    def planar_pose(self, state):
        R = state[6:15].reshape(3, 3)
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return jnp.array([state[0], state[1], yaw])

    def barrier(self, state, obstacles, *, r_safe=None):
        """Return (h, h_dot, h_ddot, A, drift), where h_dddot = A@u + drift."""
        p, v, _, _ = unpack(state)
        a, J = self.acceleration(state), self.jerk_map(state)
        rel, w = p[None, :2]-obstacles.centers, v[None, :2]-obstacles.velocities
        radius = self.radius+obstacles.radii+self.margin if r_safe is None else r_safe
        h = jnp.sum(rel*rel, axis=-1)-radius**2
        h_dot = 2*jnp.sum(rel*w, axis=1)
        h_ddot = 2*jnp.sum(w*w, axis=1)+2*rel@a[:2]
        dh_ddot_dx_g = 2*rel@J[:2]
        h_dddot_drift = 6*w@a[:2]
        return h, h_dot, h_ddot, dh_ddot_dx_g, h_dddot_drift

    def altitude_barrier(self, state):
        """Altitude derivatives in the same form as barrier()."""
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
        """Tilt/thrust h and A, where h_dot = A@u."""
        _, _, R, T = unpack(state)
        h = jnp.array([R[2, 2]-math.cos(self.tilt_max),
                       T-self.thrust_min, self.thrust_max-T])
        tilt_dh_dx_g = jnp.concatenate([(-R@skew(jnp.array([0., 0., 1.])))[2], jnp.zeros(1)])
        dh_dx_g = jnp.stack([tilt_dh_dx_g, jnp.array([0., 0., 0., 1.]),
                             jnp.array([0., 0., 0., -1.])])
        return h, dh_dx_g

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt, *, kappa=None, r_safe=None):
        """Obstacle/altitude HOCBFs and tilt/thrust CBFs."""
        check_obstacle_shapes(obstacles)
        rates = self.kappa if kappa is None else jnp.moveaxis(kappa, -1, 0)
        obs = _hocbf(*self.barrier(state, obstacles, r_safe=r_safe), kappa=rates)
        obs = masked_rows(obs.A, obs.b, obs.domain.reshape((-1, 3)), obstacles.mask)
        altitude = _hocbf(*self.altitude_barrier(state), kappa=self.kappa)
        h, dh_dx_g = self.state_barriers(state)
        b = jnp.array([self.kappa_tilt*h[0], self.kappa_thrust*h[1],
                       self.kappa_thrust*h[2]])
        return join_constraints(obs, altitude, Constraints(dh_dx_g, b, h))

    def nominal_control(self, state, goal):
        """Goal-tracking reference before safety filtering."""
        p, v, R, T = unpack(state)
        goal = jnp.concatenate([goal, jnp.array([self.altitude])])
        desired_acc = self.position_gain*(goal-p)-self.velocity_gain*v
        limits = jnp.asarray(self.acceleration_limits)
        desired_acc = jnp.clip(desired_acc, -limits, limits)
        force = self.mass*(desired_acc+jnp.array([0., 0., self.gravity]))
        bdes = force/jnp.maximum(jnp.linalg.norm(force), 1e-9)
        # Align thrust with desired force; yaw is unregulated.
        omega = self.attitude_gain*jnp.cross(jnp.array([0., 0., 1.]), R.T@bdes)
        nu = self.thrust_tracking_gain*(jnp.linalg.norm(force)-T)
        return jnp.concatenate([omega, jnp.array([nu])])

    def reward_motion(self, state, action):
        """Return horizontal forward speed and yaw rate from Rdot=R[omega]x."""
        _, velocity, rotation, _ = unpack(state)
        yaw = self.planar_pose(state)[2]
        forward = jnp.dot(velocity[:2], jnp.array([jnp.cos(yaw), jnp.sin(yaw)]))
        rotation_dot = rotation @ skew(action[:3])
        yaw_rate = ((rotation[0, 0]*rotation_dot[1, 0]-rotation[1, 0]*rotation_dot[0, 0])
                    / (rotation[0, 0]**2+rotation[1, 0]**2))
        return jnp.stack([forward, yaw_rate])
