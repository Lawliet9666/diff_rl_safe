"""DODGER planar model: [x, y, yaw, v_forward, v_lateral], [a_f, a_l, yaw_rate].

This ideal velocity-tracking model does not simulate joints or balance.
"""
from dataclasses import dataclass
import math
from typing import ClassVar

import jax.numpy as jnp

from .base import Robot, RobotStep, angle_normalize


@dataclass(frozen=True)
class HumanoidRobot(Robot):
    vf_min: float = -1.0
    vf_max: float = 2.0
    vl_min: float = -1.0
    vl_max: float = 1.0
    af_max: float = 4.0
    al_max: float = 2.0
    omega_max: float = 1.0
    state_dim: ClassVar[int] = 5
    control_dim: ClassVar[int] = 3

    def __post_init__(self):
        values = (self.vf_min, self.vf_max, self.vl_min, self.vl_max,
                  self.af_max, self.al_max, self.omega_max)
        if (not all(math.isfinite(v) for v in values)
                or not self.vf_min < 0 < self.vf_max or not self.vl_min < 0 < self.vl_max
                or min(self.af_max, self.al_max, self.omega_max) <= 0):
            raise ValueError("invalid humanoid velocity or input limits")

    def rotation(self, state):
        c, s = jnp.cos(state[2]), jnp.sin(state[2])
        return jnp.array([[c, -s], [s, c]])

    def world_velocity(self, state):
        return self.rotation(state) @ state[3:5]

    def control_affine(self, state):
        f = jnp.concatenate([self.world_velocity(state), jnp.zeros(3)])
        g = jnp.zeros((5, 3)).at[2, 2].set(1.).at[3, 0].set(1.).at[4, 1].set(1.)
        return f, g

    def dynamics(self, state, action):
        f, g = self.control_affine(state)
        return f + g @ action

    def acceleration_map(self, state):
        velocity = self.world_velocity(state)
        return jnp.column_stack([self.rotation(state), jnp.array([-velocity[1], velocity[0]])])

    def step_with_info(self, state, action, dt):
        """Midpoint world translation; exact held-input body velocities and yaw."""
        midpoint = state.at[2].add(action[2]*dt/2).at[3:5].add(action[:2]*dt/2)
        next_state = state.at[:2].add(self.world_velocity(midpoint)*dt)
        next_state = next_state.at[2].set(angle_normalize(state[2]+action[2]*dt))
        next_state = next_state.at[3:5].add(action[:2]*dt)
        return RobotStep(next_state, self.world_velocity(next_state), action)

    def input_bounds(self):
        limits = jnp.array([self.af_max, self.al_max, self.omega_max])
        return -limits, limits

    def velocity_margin(self, state):
        return jnp.array([state[3]-self.vf_min, self.vf_max-state[3],
                          state[4]-self.vl_min, self.vl_max-state[4]])

    def navigation_state(self, state):
        return state[:3]

    def control_output_matrix(self, state, control_config):
        """Physical input tracking, matching the paper's shared QP objective."""
        return jnp.eye(3)

    def nominal_control(self, state, goal, control_config):
        delta = goal-state[:2]
        desired = self.rotation(state).T @ (control_config.position_gain*delta)
        desired = jnp.clip(desired, jnp.array([self.vf_min, self.vl_min]),
                           jnp.array([self.vf_max, self.vl_max]))
        near_goal = jnp.linalg.norm(delta) < control_config.goal_tolerance
        desired = jnp.where(near_goal, 0., desired)
        yaw_error = angle_normalize(jnp.arctan2(delta[1], delta[0])-state[2])
        omega = jnp.where(near_goal, 0., jnp.clip(control_config.heading_gain*yaw_error,
                                                -self.omega_max, self.omega_max))
        acceleration = control_config.velocity_gain*(desired-state[3:5])
        acceleration -= omega*jnp.array([-state[4], state[3]])
        return jnp.concatenate([acceleration, jnp.array([omega])])
