"""Unicycle equations and the original SocialNav Euler/clipping convention."""
from dataclasses import dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Robot, RobotStep, angle_normalize, check_obstacle_shapes, masked_rows, _control_parameters


@dataclass(frozen=True)
class UnicycleRobot(Robot):
    v_max: float
    omega_max: float
    state_dim: ClassVar[int] = 3
    control_dim: ClassVar[int] = 2

    # Collision and navigation.
    radius: float
    margin: float
    goal_tolerance: float

    # Nominal tracking; lookahead is shared with the CBF and QP.
    lookahead: float
    position_gain: float

    # Linear class-K rate: alpha_1(s) = kappa[0]*s.
    kappa: tuple

    # QP tracking weights.
    weights: tuple

    @classmethod
    def from_config(cls, config):
        """Extract this robot's parameters from the complete Hydra configuration."""
        r = config.robot
        parameters = _control_parameters(
            config.controller.unicycle, ("lookahead", "position_gain", "kappa", "weights"),
            "controller.unicycle")
        parameters.update(v_max=r.v_max, omega_max=r.omega_max, radius=r.radius,
                          margin=float(config.env.safety_margin),
                          goal_tolerance=float(config.env.robot_goal.tolerance))
        for name in ("kappa", "weights"):
            parameters[name] = tuple(parameters[name])
        return cls(**parameters)

    def __post_init__(self):
        vals = (self.radius, self.lookahead, *self.kappa, self.v_max, self.omega_max,
                self.position_gain, self.goal_tolerance, *self.weights)
        if (not all(math.isfinite(v) and v > 0 for v in vals)
                or not math.isfinite(self.margin) or self.margin < 0
                or len(self.kappa) != 1 or len(self.weights) != 2):
            raise ValueError("radii, lookahead, gains and limits must be positive")

    def dynamics(self, state, action):
        v, omega = action
        return jnp.array([v*jnp.cos(state[2]), v*jnp.sin(state[2]), omega])

    def step(self, state, action, dt):
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

    def initial_state(self, pose):
        return jnp.asarray(pose, dtype=jnp.float32)

    def input_bounds(self):
        limits = jnp.array([self.v_max, self.omega_max])
        return -limits, limits

    def planar_pose(self, state):
        return state

    def control_output_matrix(self, state):
        """J maps [v, omega] to the world velocity of the lookahead point."""
        c, s = jnp.cos(state[2]), jnp.sin(state[2])
        ell = self.lookahead
        return jnp.array([[c, -ell*s], [s, ell*c]])

    def barrier(self, state, obstacles):
        """Return h, dh_dx_g and obstacle_motion; h_dot = dh_dx_g@u-obstacle_motion.

        dh_dx_g is the state gradient of h multiplied by the input map g.
        """
        heading = jnp.array([jnp.cos(state[2]), jnp.sin(state[2])])
        q = state[:2]+self.lookahead*heading
        rel = q-obstacles.centers
        # Inflating by lookahead also separates the actual robot disk.
        distance = self.radius+obstacles.radii+self.margin+self.lookahead
        h = jnp.sum(rel*rel, axis=-1)-distance**2
        dh_dx_g = 2*rel@self.control_output_matrix(state)
        obstacle_motion = 2*jnp.sum(rel*obstacles.velocities, axis=1)
        return h, dh_dx_g, obstacle_motion

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt):
        """First-order obstacle CBFs for the unicycle lookahead point."""
        check_obstacle_shapes(obstacles)
        h, dh_dx_g, obstacle_motion = self.barrier(state, obstacles)
        b = self.kappa[0]*h-obstacle_motion
        return masked_rows(dh_dx_g, b, h[:, None], obstacles.mask)

    def nominal_control(self, state, goal):
        """Represent the old planar go-to-goal velocity in physical input coordinates."""
        delta = goal-state[:2]
        velocity = self.position_gain*delta
        speed = jnp.linalg.norm(velocity)
        velocity *= jnp.minimum(1., self.v_max/jnp.maximum(speed, 1e-9))
        nominal = jnp.linalg.solve(self.control_output_matrix(state), velocity)
        return jnp.where(jnp.linalg.norm(delta) < self.goal_tolerance, jnp.zeros(2), nominal)
