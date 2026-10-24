"""DODGER planar model: [x, y, yaw, v_forward, v_lateral], [a_f, a_l, yaw_rate].

This ideal velocity-tracking model does not simulate joints or balance.
"""
from dataclasses import KW_ONLY, dataclass
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, angle_normalize, Constraints,
                   check_obstacle_shapes, masked_rows, join_constraints)


def _parabolic_value(z, w, radius, cfg):
    distance = jnp.linalg.norm(z)
    e = z / distance
    tangent = jnp.array([-e[1], e[0]])
    clearance = jnp.sqrt(z @ z - radius**2)
    lam = cfg.parabolic_gain * clearance / jnp.sqrt(w @ w + cfg.speed_epsilon**2)
    return e @ w + lam * (tangent @ w)**2 + cfg.clearance_gain * clearance


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

    # Optional settings; ordinary environment construction needs only the physical fields above.
    _: KW_ONLY

    # Collision and navigation.
    radius: float = 0.3
    margin: float = 0.05
    goal_tolerance: float = 0.6

    # Nominal tracking.
    position_gain: float = 1.0
    velocity_gain: float = 2.0
    heading_gain: float = 2.0

    # CBF choice, shape and decay.
    barrier: str = 'distance'
    gains: tuple = (1.5, 1.5)
    alpha: float = 2.0
    parabolic_gain: float = 1.0
    clearance_gain: float = 1.0
    speed_epsilon: float = 0.05

    # QP tracking weights.
    weights: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        values = (self.vf_min, self.vf_max, self.vl_min, self.vl_max,
                  self.af_max, self.al_max, self.omega_max)
        if (not all(math.isfinite(v) for v in values)
                or not self.vf_min < 0 < self.vf_max or not self.vl_min < 0 < self.vl_max
                or min(self.af_max, self.al_max, self.omega_max) <= 0):
            raise ValueError("invalid humanoid velocity or input limits")

        values = (self.goal_tolerance, self.radius, *self.gains,
                  self.alpha, self.parabolic_gain, self.clearance_gain,
                  self.speed_epsilon, *self.weights, self.position_gain,
                  self.velocity_gain, self.heading_gain)
        if (self.barrier not in {"distance", "parabolic"}
                or len(self.gains) != 2 or len(self.weights) != 3
                or not all(math.isfinite(v) and v > 0 for v in values)
                or not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError("invalid humanoid barrier, gains or controller limits")

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

    def control_output_matrix(self, state):
        """Physical input tracking, matching the paper's shared QP objective."""
        return jnp.eye(3)

    def barrier_domain_valid(self, state, obstacles):
        """The parabolic barrier is defined strictly outside inflated obstacles."""
        if self.barrier == "distance":
            return jnp.array(True)
        radius = self.radius + obstacles.radii + self.margin
        q = jnp.sum((state[:2]-obstacles.centers)**2, axis=1) - radius**2
        return jnp.all((obstacles.mask <= .5) | (q > 0))

    def distance_barrier(self, state, obstacles):
        """Return h, h_dot, dh_dot_dx_g and h_ddot_drift.

        dh_dot_dx_g = (d h_dot / d state) @ g(state).
        h_ddot = dh_dot_dx_g@u+h_ddot_drift.
        """
        z = state[:2]-obstacles.centers
        w = self.world_velocity(state)-obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin
        B = self.acceleration_map(state)
        h = jnp.sum(z*z, axis=1)-radius**2
        h_dot = 2*jnp.sum(z*w, axis=1)
        dh_dot_dx_g = 2*z @ B
        h_ddot_drift = 2*jnp.sum(w*w, axis=1)
        return h, h_dot, dh_dot_dx_g, h_ddot_drift

    def parabolic_barrier(self, state, obstacles):
        """Return h, squared-clearance q, dh_dx_g and h_dot_drift.

        dh_dz and dh_dw are gradients with respect to relative position/velocity.
        h_dot = dh_dx_g@u+h_dot_drift.
        """
        z = state[:2]-obstacles.centers
        w = self.world_velocity(state)-obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin
        B = self.acceleration_map(state)
        # Give inactive padding a valid point before differentiating the square root.
        dummy = jnp.stack([radius+1., jnp.zeros_like(radius)], axis=1)
        z = jnp.where((obstacles.mask > .5)[:, None], z, dummy)
        w = jnp.where((obstacles.mask > .5)[:, None], w, 0.)
        value = lambda z, w, r: _parabolic_value(z, w, r, self)
        h = jax.vmap(value)(z, w, radius)
        dh_dz, dh_dw = jax.vmap(jax.grad(value, argnums=(0, 1)))(z, w, radius)
        q = jnp.sum(z*z, axis=1)-radius**2
        dh_dx_g = dh_dw @ B
        h_dot_drift = jnp.sum(dh_dz*w, axis=1)
        return h, q, dh_dx_g, h_dot_drift

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt):
        """Assemble distance/parabolic obstacle CBFs and next-step velocity bounds."""
        check_obstacle_shapes(obstacles)
        if self.barrier == "distance":
            h, h_dot, A, h_ddot_drift = self.distance_barrier(state, obstacles)
            k1, k2 = self.gains
            b = h_ddot_drift+(k1+k2)*h_dot+k1*k2*h
            domain = jnp.stack([h, h_dot+k1*h], axis=1)
        elif self.barrier == "parabolic":
            h, q, A, h_dot_drift = self.parabolic_barrier(state, obstacles)
            b = h_dot_drift+self.alpha*h
            domain = jnp.stack([h, q], axis=1)
        else:
            raise ValueError("humanoid barrier must be distance or parabolic")
        obstacle_rows = masked_rows(A, b, domain, obstacles.mask)
        # Bound the ideal model's next body-velocity command without integrator clipping.
        margin = self.velocity_margin(state)
        velocity_rows = Constraints(
            dt*jnp.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]),
            margin, margin)
        return join_constraints(obstacle_rows, velocity_rows)

    def nominal_control(self, state, goal):
        delta = goal-state[:2]
        desired = self.rotation(state).T @ (self.position_gain*delta)
        desired = jnp.clip(desired, jnp.array([self.vf_min, self.vl_min]),
                           jnp.array([self.vf_max, self.vl_max]))
        near_goal = jnp.linalg.norm(delta) < self.goal_tolerance
        desired = jnp.where(near_goal, 0., desired)
        yaw_error = angle_normalize(jnp.arctan2(delta[1], delta[0])-state[2])
        omega = jnp.where(near_goal, 0., jnp.clip(self.heading_gain*yaw_error,
                                                -self.omega_max, self.omega_max))
        acceleration = self.velocity_gain*(desired-state[3:5])
        acceleration -= omega*jnp.array([-state[4], state[3]])
        return jnp.concatenate([acceleration, jnp.array([omega])])
