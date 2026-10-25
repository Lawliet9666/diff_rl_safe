"""Planar velocity model with state [x, y, 0]."""
from dataclasses import dataclass
import math
from typing import ClassVar

import jax.numpy as jnp

from .base import Robot, RobotStep, check_obstacle_shapes, masked_rows, _control_parameters


@dataclass(frozen=True)
class SingleIntegratorRobot(Robot):
    v_max: float
    radius: float
    margin: float
    kappa: tuple
    weights: tuple
    state_dim: ClassVar[int] = 3
    control_dim: ClassVar[int] = 2
    policy_feature_dim: ClassVar[int] = 2

    @classmethod
    def from_config(cls, config):
        values = _control_parameters(config.controller.single_integrator, ("kappa", "weights"),
                                     "controller.single_integrator")
        return cls(v_max=config.robot.v_max, radius=config.robot.radius,
                   margin=float(config.env.safety_margin),
                   kappa=tuple(values['kappa']), weights=tuple(values['weights']))

    def __post_init__(self):
        if (len(self.kappa) != 1 or len(self.weights) != 2
                or not all(math.isfinite(v) and v > 0 for v in
                           (self.v_max, self.radius, *self.kappa, *self.weights))
                or not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError("invalid single-integrator CBF parameters")

    def dynamics(self, state, action):
        return jnp.array([action[0], action[1], 0.])

    def step(self, state, action, dt):
        action = jnp.asarray(action, dtype=jnp.float32)
        speed = jnp.linalg.norm(action, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.v_max / jnp.maximum(speed, 1e-8))
        applied = action * scale
        next_xy = state[..., 0:2] + applied * dt
        next_theta = jnp.zeros_like(state[..., 2:3])
        next_state = jnp.concatenate([next_xy, next_theta], axis=-1)
        return RobotStep(next_state, applied, applied)

    def initial_state(self, pose):
        return jnp.asarray(pose, dtype=jnp.float32)

    def policy_features(self, state, velocity, *, position_scale, velocity_scale):
        """Measured planar velocity in the model's world-aligned frame."""
        return velocity / velocity_scale

    def input_bounds(self):
        # step() also enforces the Euclidean speed limit.
        limits = jnp.full(2, self.v_max)
        return -limits, limits

    def planar_pose(self, state):
        return state

    def control_output_matrix(self, state):
        """Track world-frame planar velocity."""
        return jnp.eye(2, dtype=state.dtype)

    def cbf_constraints(self, state, obstacles, dt):
        """First-order distance CBFs for the physical input [vx, vy]."""
        check_obstacle_shapes(obstacles)
        relative = state[:2]-obstacles.centers
        radius = self.radius+obstacles.radii+self.margin
        h = jnp.sum(relative**2, axis=1)-radius**2
        b = self.kappa[0]*h-2*jnp.sum(relative*obstacles.velocities, axis=1)
        return masked_rows(2*relative, b, h[:, None], obstacles.mask)
