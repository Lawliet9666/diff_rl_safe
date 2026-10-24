"""Unicycle equations and the original SocialNav Euler/clipping convention."""
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Robot, RobotStep, angle_normalize, check_obstacle_shapes, masked_rows


@dataclass(frozen=True)
class UnicycleRobot(Robot):
    v_max: float = 1.5
    omega_max: float = 1.57
    state_dim: ClassVar[int] = 3
    control_dim: ClassVar[int] = 2

    def dynamics(self, state, action):
        v, omega = action
        return jnp.array([v*jnp.cos(state[2]), v*jnp.sin(state[2]), omega])

    def step_with_info(self, state, action, dt):
        action = jnp.asarray(action, dtype=jnp.float32)
        v = jnp.clip(action[..., 0], -self.v_max, self.v_max)
        omega = jnp.clip(action[..., 1], -self.omega_max, self.omega_max)
        theta = state[..., 2]
        next_theta = angle_normalize(theta + omega * dt)
        next_state = jnp.stack(
            [state[..., 0] + v * jnp.cos(theta) * dt,
             state[..., 1] + v * jnp.sin(theta) * dt,
             next_theta], axis=-1)
        applied = jnp.stack([v, omega], axis=-1)
        velocity = jnp.stack([v * jnp.cos(next_theta), v * jnp.sin(next_theta)], axis=-1)
        return RobotStep(next_state, velocity, applied)

    def input_bounds(self):
        limits = jnp.array([self.v_max, self.omega_max])
        return -limits, limits

    def navigation_state(self, state):
        return state

    def control_output_matrix(self, state, control_config):
        """J maps [v, omega] to the world velocity of the lookahead point."""
        c, s = jnp.cos(state[2]), jnp.sin(state[2])
        ell = control_config.lookahead
        return jnp.array([[c, -ell*s], [s, ell*c]])

    def barrier(self, state, obstacles, cfg):
        """Return h, dh_dx_g and obstacle_motion; h_dot = dh_dx_g@u-obstacle_motion.

        dh_dx_g is the state gradient of h multiplied by the input map g.
        """
        heading = jnp.array([jnp.cos(state[2]), jnp.sin(state[2])])
        q = state[:2]+cfg.lookahead*heading
        rel = q-obstacles.centers
        # Inflating by lookahead also separates the actual robot disk.
        distance = cfg.radius+obstacles.radii+cfg.margin+cfg.lookahead
        h = jnp.sum(rel*rel, axis=-1)-distance**2
        dh_dx_g = 2*rel@self.control_output_matrix(state, cfg)
        obstacle_motion = 2*jnp.sum(rel*obstacles.velocities, axis=1)
        return h, dh_dx_g, obstacle_motion

    @partial(jax.jit, static_argnames=("self", "cbf_config"))
    def cbf_constraints(self, state, obstacles, cbf_config):
        """First-order obstacle CBFs for the unicycle lookahead point."""
        check_obstacle_shapes(obstacles)
        h, dh_dx_g, obstacle_motion = self.barrier(state, obstacles, cbf_config)
        b = cbf_config.alpha*h-obstacle_motion
        return masked_rows(dh_dx_g, b, h[:, None], obstacles.mask)

    def nominal_control(self, state, goal, control_config):
        """Represent the old planar go-to-goal velocity in physical input coordinates."""
        delta = goal-state[:2]
        velocity = control_config.position_gain*delta
        speed = jnp.linalg.norm(velocity)
        velocity *= jnp.minimum(1., self.v_max/jnp.maximum(speed, 1e-9))
        nominal = jnp.linalg.solve(self.control_output_matrix(state, control_config), velocity)
        return jnp.where(jnp.linalg.norm(delta) < control_config.goal_tolerance, jnp.zeros(2), nominal)
