"""Planar double integrator: state [p_x, p_y, v_x, v_y], input [a_x, a_y].

Positions, velocities and accelerations are expressed in the world frame.
Integration preserves the commanded acceleration; limits are checked separately.
"""
from dataclasses import dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, Constraints, check_obstacle_shapes,
                   masked_rows, join_constraints, _control_parameters)


@dataclass(frozen=True)
class DoubleIntegratorRobot(Robot):
    v_max: float
    a_max: float
    radius: float
    margin: float
    goal_tolerance: float
    position_gain: float
    velocity_gain: float
    gains: tuple
    speed_facets: int
    weights: tuple
    state_dim: ClassVar[int] = 4
    control_dim: ClassVar[int] = 2

    @classmethod
    def from_config(cls, config):
        """Read physical, navigation and CBF-QP parameters from Hydra."""
        parameters = _control_parameters(
            config.controller.double_integrator,
            ("position_gain", "velocity_gain", "gains", "speed_facets", "weights"),
            "controller.double_integrator")
        parameters.update(v_max=config.robot.v_max, a_max=config.robot.a_max,
                          radius=config.robot.radius, margin=float(config.env.safety_margin),
                          goal_tolerance=float(config.env.robot_goal.tolerance))
        for name in ("gains", "weights"):
            parameters[name] = tuple(parameters[name])
        return cls(**parameters)

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0 for value in (self.v_max, self.a_max)):
            raise ValueError("double-integrator speed and acceleration limits must be finite and positive")
        values = (self.radius, self.goal_tolerance, self.position_gain,
                  self.velocity_gain, *self.gains, *self.weights)
        if (len(self.gains) != 2 or len(self.weights) != 2
                or not all(math.isfinite(value) and value > 0 for value in values)
                or not math.isfinite(self.margin) or self.margin < 0
                or not isinstance(self.speed_facets, int) or isinstance(self.speed_facets, bool)
                or self.speed_facets < 4 or self.speed_facets % 2):
            raise ValueError("invalid double-integrator CBF gains, weights or speed polygon")

    def control_affine(self, state):
        f = jnp.concatenate([state[2:], jnp.zeros_like(state[2:])])
        g = jnp.concatenate([jnp.zeros((2, 2), dtype=state.dtype),
                             jnp.eye(2, dtype=state.dtype)], axis=0)
        return f, g

    def dynamics(self, state, action):
        return jnp.concatenate([state[2:], jnp.asarray(action, dtype=state.dtype)])

    def step(self, state, action, dt):
        """Exact zero-order-hold integration for a constant acceleration input."""
        action = jnp.asarray(action, dtype=state.dtype)
        position = state[:2] + state[2:]*dt + .5*action*dt**2
        velocity = state[2:] + action*dt
        return RobotStep(jnp.concatenate([position, velocity]), velocity, action)

    def initial_state(self, pose):
        pose = jnp.asarray(pose, dtype=float)
        return jnp.concatenate([pose[:2], jnp.zeros_like(pose[:2])])

    def check_state_limits(self, state, tolerance=1e-6):
        return (jnp.all(jnp.isfinite(state))
                & (jnp.linalg.norm(state[2:]) <= self.v_max + tolerance))

    def input_bounds(self):
        limits = jnp.full(2, self.a_max)
        return -limits, limits

    def planar_pose(self, state):
        """The holonomic point model has no heading; use zero for its projection."""
        return jnp.concatenate([state[:2], jnp.zeros_like(state[:1])])

    def control_output_matrix(self, state):
        """Track world-frame acceleration with the common QP objective."""
        return jnp.eye(2, dtype=state.dtype)

    def distance_barrier(self, state, obstacles):
        """Distance derivatives with constant obstacle velocity during prediction."""
        relative = state[:2] - obstacles.centers
        velocity = state[2:] - obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin
        h = jnp.sum(relative**2, axis=1) - radius**2
        h_dot = 2*jnp.sum(relative*velocity, axis=1)
        h_ddot_drift = 2*jnp.sum(velocity**2, axis=1)
        return h, h_dot, 2*relative, h_ddot_drift

    def _speed_normals(self):
        angles = (2*jnp.arange(self.speed_facets) + 1)*jnp.pi/self.speed_facets
        return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt):
        """Second-order obstacle CBFs and next-velocity polygon constraints.

        The regular polygon is inscribed in the Euclidean speed disk. Both
        endpoint velocities inside it imply the held-input velocity remains
        inside the disk throughout the step; no post-QP clipping is applied.
        Obstacle CBFs are evaluated at control instants, as for the other robots.
        """
        check_obstacle_shapes(obstacles)
        h, h_dot, A, drift = self.distance_barrier(state, obstacles)
        k1, k2 = self.gains
        b = drift + (k1+k2)*h_dot + k1*k2*h
        domain = jnp.stack([h, h_dot+k1*h], axis=1)
        obstacle_rows = masked_rows(A, b, domain, obstacles.mask)
        normals = self._speed_normals()
        margin = self.v_max*jnp.cos(jnp.pi/self.speed_facets) - normals@state[2:]
        speed_rows = Constraints(-dt*normals, margin, margin)
        return join_constraints(obstacle_rows, speed_rows)

    def nominal_control(self, state, goal):
        desired = self.position_gain*(goal-state[:2])
        speed = jnp.linalg.norm(desired)
        desired *= jnp.minimum(1., self.v_max/jnp.maximum(speed, 1e-9))
        desired = jnp.where(jnp.linalg.norm(goal-state[:2]) < self.goal_tolerance, 0., desired)
        return self.velocity_gain*(desired-state[2:])
