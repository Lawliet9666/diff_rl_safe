"""Stateless robot contract; all changing state stays in JAX arrays."""
from abc import ABC, abstractmethod
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp


class RobotStep(NamedTuple):
    state: Array
    velocity: Array
    action: Array


def angle_normalize(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


class Robot(ABC):
    state_dim: int
    control_dim: int

    @abstractmethod
    def dynamics(self, state, action):
        """Continuous-time derivative for one robot state."""
        raise NotImplementedError

    @abstractmethod
    def step_with_info(self, state, action, dt):
        """Return next state, world velocity and applied physical input."""
        raise NotImplementedError

    def step(self, state, action, dt):
        return self.step_with_info(state, action, dt).state

    @abstractmethod
    def input_bounds(self):
        """Componentwise physical input bounds, as (lower, upper)."""
        raise NotImplementedError

    @abstractmethod
    def navigation_state(self, state):
        """Return the [x, y, yaw] projection used by the crowd and renderer."""
        raise NotImplementedError

    def nominal_control(self, state, goal, control_config):
        """Return a goal-tracking reference without applying a safety filter."""
        raise NotImplementedError(
            f"Nominal control is not implemented for {type(self).__name__}")

    def control_output_matrix(self, state, control_config):
        """Map physical control to the motion output tracked by the controller."""
        raise NotImplementedError(
            f"Control output mapping is not implemented for {type(self).__name__}")
