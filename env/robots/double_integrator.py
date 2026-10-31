"""World-frame double integrator: state [px, py, vx, vy], input [ax, ay]."""
from dataclasses import dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, Constraints, check_obstacle_shapes, masked_rows,
                   join_constraints, goal_velocity, _control_parameters)


@dataclass(frozen=True)
class DoubleIntegratorRobot(Robot):
    a_max: float
    v_max: float
    radius: float
    margin: float
    goal_tolerance: float
    position_gain: float
    velocity_gain: float
    kappa: tuple  # HOCBF rates [kappa_1, kappa_2].
    kappa_speed: float  # First-order total-speed CBF rate.
    weights: tuple
    state_dim: ClassVar[int] = 4
    control_dim: ClassVar[int] = 2
    policy_feature_dim: ClassVar[int] = 2

    @classmethod
    def from_config(cls, config):
        parameters = _control_parameters(
            config.controller.double_integrator,
            ("position_gain", "velocity_gain", "kappa", "kappa_speed", "weights"),
            "controller.double_integrator")
        parameters.update(a_max=config.robot.a_max, v_max=config.robot.v_max,
                          radius=config.robot.radius, margin=float(config.env.safety_margin),
                          goal_tolerance=float(config.env.robot_goal.tolerance))
        for name in ("kappa", "weights"):
            parameters[name] = tuple(parameters[name])
        return cls(**parameters)

    def __post_init__(self):
        if not math.isfinite(self.a_max) or self.a_max <= 0:
            raise ValueError("double-integrator acceleration limit must be finite and positive")
        values = (self.v_max, self.radius, self.goal_tolerance, self.position_gain,
                  self.velocity_gain, self.kappa_speed, *self.kappa, *self.weights)
        if (len(self.kappa) != 2 or len(self.weights) != 2
                or not all(math.isfinite(value) and value > 0 for value in values)
                or not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError("invalid double-integrator speed limit, CBF gains or weights")

    def control_affine(self, state):
        f = jnp.concatenate([state[2:], jnp.zeros_like(state[2:])])
        g = jnp.concatenate([jnp.zeros((2, 2), dtype=state.dtype),
                             jnp.eye(2, dtype=state.dtype)], axis=0)
        return f, g

    def dynamics(self, state, action):
        return jnp.concatenate([state[2:], jnp.asarray(action, dtype=state.dtype)])

    def step(self, state, action, dt):
        """Exact integration under constant acceleration."""
        action = jnp.asarray(action, dtype=state.dtype)
        position = state[:2] + state[2:]*dt + .5*action*dt**2
        velocity = state[2:] + action*dt
        return RobotStep(jnp.concatenate([position, velocity]), velocity, action)

    def initial_state(self, pose):
        pose = jnp.asarray(pose, dtype=float)
        return jnp.concatenate([pose[:2], jnp.zeros_like(pose[:2])])

    def check_state_limits(self, state, tolerance=1e-6):
        return (jnp.all(jnp.isfinite(state))
                & (jnp.linalg.norm(state[2:]) <= self.v_max+tolerance))

    def velocity_constraints(self, state):
        """Continuous-time speed CBF; held-step speed bounds need separate checks."""
        velocity = state[2:]
        h = self.v_max**2 - velocity @ velocity
        return Constraints(-2*velocity[None, :], self.kappa_speed*h[None], h[None])

    def policy_features(self, state, velocity, *, position_scale, velocity_scale):
        """World-frame velocity; this model has no heading."""
        return state[2:4] / velocity_scale

    def input_bounds(self):
        limits = jnp.full(2, self.a_max)
        return -limits, limits

    def planar_pose(self, state):
        """Return [x, y, 0]; this model has no heading."""
        return jnp.concatenate([state[:2], jnp.zeros_like(state[:1])])

    def control_output_matrix(self, state):
        """Track world-frame acceleration."""
        return jnp.eye(2, dtype=state.dtype)

    def distance_barrier(self, state, obstacles, *, r_safe=None):
        """Distance derivatives assuming constant obstacle velocity."""
        relative = state[:2] - obstacles.centers
        velocity = state[2:] - obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin if r_safe is None else r_safe
        h = jnp.sum(relative**2, axis=1) - radius**2
        h_dot = 2*jnp.sum(relative*velocity, axis=1)
        h_ddot_drift = 2*jnp.sum(velocity**2, axis=1)
        return h, h_dot, 2*relative, h_ddot_drift

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt, *, kappa=None, r_safe=None):
        """Obstacle HOCBFs followed by a hard continuous-time speed CBF."""
        check_obstacle_shapes(obstacles)
        h, h_dot, A, drift = self.distance_barrier(state, obstacles, r_safe=r_safe)
        kappa1, kappa2 = self.kappa if kappa is None else jnp.moveaxis(kappa, -1, 0)
        b = drift + (kappa1+kappa2)*h_dot + kappa1*kappa2*h
        domain = jnp.stack([h, h_dot+kappa1*h], axis=1)
        obstacle_rows = masked_rows(A, b, domain, obstacles.mask)
        return join_constraints(obstacle_rows, self.velocity_constraints(state))

    def nominal_control(self, state, goal):
        desired = goal_velocity(state[:2], goal, self.position_gain,
                                self.v_max, self.goal_tolerance)
        return self.velocity_gain*(desired-state[2:])

    def reward_motion(self, state, action):
        """Return speed and zero yaw rate."""
        return jnp.array([jnp.linalg.norm(state[2:]), 0.], dtype=state.dtype)
