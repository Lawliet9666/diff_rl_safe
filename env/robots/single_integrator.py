"""Planar velocity model with the original three-coordinate storage layout."""
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from .base import Robot, RobotStep


@dataclass(frozen=True)
class SingleIntegratorRobot(Robot):
    v_max: float = 1.5
    state_dim: ClassVar[int] = 3
    control_dim: ClassVar[int] = 2

    def dynamics(self, state, action):
        return jnp.array([action[0], action[1], 0.])

    def step_with_info(self, state, action, dt):
        action = jnp.asarray(action, dtype=jnp.float32)
        speed = jnp.linalg.norm(action, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.v_max / jnp.maximum(speed, 1e-8))
        applied = action * scale
        next_xy = state[..., 0:2] + applied * dt
        next_theta = jnp.zeros_like(state[..., 2:3])
        next_state = jnp.concatenate([next_xy, next_theta], axis=-1)
        return RobotStep(next_state, applied, applied)

    def input_bounds(self):
        # The step also enforces the original Euclidean speed limit.
        limits = jnp.full(2, self.v_max)
        return -limits, limits

    def navigation_state(self, state):
        return state
