"""DODGER reduced model: state [x, y, yaw, vf, vl], input [af, al, yaw_rate]."""
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import partial
import math
from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import (Robot, RobotStep, Constraints, angle_normalize, goal_velocity,
                   check_obstacle_shapes, masked_rows, join_constraints,
                   _control_parameters)


def _parabolic_value(z, w, radius, cfg):
    distance = jnp.linalg.norm(z)
    e = z / distance
    tangent = jnp.array([-e[1], e[0]])
    clearance = jnp.sqrt(z @ z - radius**2)
    lam = cfg.parabolic_gain * clearance / jnp.sqrt(w @ w + cfg.speed_epsilon**2)
    return e @ w + lam * (tangent @ w)**2 + cfg.clearance_gain * clearance


@dataclass(frozen=True)
class DistanceCBFParams:
    kappa: tuple  # HOCBF rates [kappa_1, kappa_2].

    def __post_init__(self):
        if len(self.kappa) != 2 or not all(math.isfinite(v) and v > 0 for v in self.kappa):
            raise ValueError("invalid humanoid distance CBF gains")


@dataclass(frozen=True)
class ParabolicCBFParams:
    kappa: tuple  # CBF rate [kappa_1].
    parabolic_gain: float
    clearance_gain: float
    speed_epsilon: float

    def __post_init__(self):
        values = (*self.kappa, self.parabolic_gain, self.clearance_gain, self.speed_epsilon)
        if len(self.kappa) != 1 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("invalid humanoid parabolic CBF parameters")


@dataclass(frozen=True)
class HumanoidRobot(Robot):
    af_max: float
    al_max: float
    omega_max: float
    v_max: float
    state_dim: ClassVar[int] = 5
    control_dim: ClassVar[int] = 3
    policy_feature_dim: ClassVar[int] = 2

    radius: float
    margin: float
    goal_tolerance: float

    position_gain: float
    velocity_gain: float
    heading_gain: float

    barrier: str
    cbf: DistanceCBFParams | ParabolicCBFParams
    kappa_speed: float

    weights: tuple

    @classmethod
    def from_config(cls, config):
        r = config.robot
        section = config.controller.humanoid
        barrier = section.get("barrier") if isinstance(section, Mapping) else None
        if barrier == "distance":
            cbf_type = DistanceCBFParams
        elif barrier == "parabolic":
            cbf_type = ParabolicCBFParams
        else:
            raise ValueError("humanoid barrier must be distance or parabolic")
        cbf_names = tuple(field.name for field in fields(cbf_type))
        parameters = _control_parameters(
            section,
            ("barrier", "position_gain", "velocity_gain", "heading_gain", "kappa_speed", "weights", *cbf_names),
            "controller.humanoid", selected_cbf=barrier)
        parameters["kappa"] = tuple(parameters["kappa"])
        parameters["cbf"] = cbf_type(**{name: parameters.pop(name) for name in cbf_names})
        parameters.update(af_max=r.af_max, al_max=r.al_max,
                          omega_max=r.omega_max, v_max=r.v_max, radius=r.radius,
                          margin=float(config.env.safety_margin),
                          goal_tolerance=float(config.env.robot_goal.tolerance))
        parameters["weights"] = tuple(parameters["weights"])
        return cls(**parameters)

    def __post_init__(self):
        values = (self.af_max, self.al_max, self.omega_max, self.v_max, self.kappa_speed)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("invalid humanoid velocity, input limits or speed CBF gain")

        valid_cbf = ((self.barrier == "distance" and isinstance(self.cbf, DistanceCBFParams))
                     or (self.barrier == "parabolic" and isinstance(self.cbf, ParabolicCBFParams)))
        values = (self.goal_tolerance, self.radius, *self.weights,
                  self.position_gain, self.velocity_gain, self.heading_gain)
        if (not valid_cbf or len(self.weights) != 3
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

    def step(self, state, action, dt):
        """Midpoint translation with exact held-input body velocities and yaw."""
        midpoint = state.at[2].add(action[2]*dt/2).at[3:5].add(action[:2]*dt/2)
        next_state = state.at[:2].add(self.world_velocity(midpoint)*dt)
        next_state = next_state.at[2].set(angle_normalize(state[2]+action[2]*dt))
        next_state = next_state.at[3:5].add(action[:2]*dt)
        return RobotStep(next_state, self.world_velocity(next_state), action)

    def initial_state(self, pose):
        return jnp.concatenate([pose.astype(jnp.float64), jnp.zeros(2)])

    def check_state_limits(self, state, tolerance=1e-6):
        return (jnp.all(jnp.isfinite(state))
                & (jnp.linalg.norm(state[3:5]) <= self.v_max+tolerance))

    def physical_constraints(self, state, dt):
        return self.velocity_constraints(state)

    def velocity_constraints(self, state):
        """Continuous-time speed CBF; held-step speed bounds need separate checks."""
        velocity = state[3:5]
        h = self.v_max**2-velocity@velocity
        A = jnp.concatenate([-2*velocity, jnp.zeros(1, dtype=state.dtype)])[None, :]
        return Constraints(A, self.kappa_speed*h[None], h[None])

    def policy_features(self, state, velocity, *, position_scale, velocity_scale):
        """Body-frame forward/lateral velocities, without a duplicate world velocity."""
        return state[3:5] / velocity_scale

    def input_bounds(self):
        limits = jnp.array([self.af_max, self.al_max, self.omega_max])
        return -limits, limits

    def planar_pose(self, state):
        return state[:3]

    def control_output_matrix(self, state):
        """Track physical inputs."""
        return jnp.eye(3)

    def validate_cbf_domain(self, state, obstacles):
        """Parabolic CBF requires strict separation from inflated obstacles."""
        if self.barrier == "parabolic":
            radius = self.radius + obstacles.radii + self.margin
            q = jnp.sum((state[:2]-obstacles.centers)**2, axis=1) - radius**2
            if not bool(jnp.all((obstacles.mask <= .5) | (q > 0))):
                raise ValueError("parabolic CBF requires strict separation from inflated obstacles")

    def distance_barrier(self, state, obstacles, *, r_safe=None):
        """Return (h, h_dot, A, drift), where h_ddot = A@u + drift."""
        z = state[:2]-obstacles.centers
        w = self.world_velocity(state)-obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin if r_safe is None else r_safe
        B = self.acceleration_map(state)
        h = jnp.sum(z*z, axis=1)-radius**2
        h_dot = 2*jnp.sum(z*w, axis=1)
        dh_dot_dx_g = 2*z @ B
        h_ddot_drift = 2*jnp.sum(w*w, axis=1)
        return h, h_dot, dh_dot_dx_g, h_ddot_drift

    def parabolic_barrier(self, state, obstacles):
        """Return (h, squared_clearance, A, drift), where h_dot = A@u + drift."""
        z = state[:2]-obstacles.centers
        w = self.world_velocity(state)-obstacles.velocities
        radius = self.radius + obstacles.radii + self.margin
        B = self.acceleration_map(state)
        # Keep sqrt derivatives finite for inactive rows.
        dummy = jnp.stack([radius+1., jnp.zeros_like(radius)], axis=1)
        z = jnp.where((obstacles.mask > .5)[:, None], z, dummy)
        w = jnp.where((obstacles.mask > .5)[:, None], w, 0.)
        value = lambda z, w, r: _parabolic_value(z, w, r, self.cbf)
        h = jax.vmap(value)(z, w, radius)
        dh_dz, dh_dw = jax.vmap(jax.grad(value, argnums=(0, 1)))(z, w, radius)
        q = jnp.sum(z*z, axis=1)-radius**2
        dh_dx_g = dh_dw @ B
        h_dot_drift = jnp.sum(dh_dz*w, axis=1)
        return h, q, dh_dx_g, h_dot_drift

    @partial(jax.jit, static_argnames=("self",))
    def cbf_constraints(self, state, obstacles, dt, *, kappa=None, r_safe=None):
        """Obstacle CBFs followed by a hard continuous-time speed CBF."""
        check_obstacle_shapes(obstacles)
        if self.barrier == "distance":
            h, h_dot, A, h_ddot_drift = self.distance_barrier(state, obstacles, r_safe=r_safe)
            kappa1, kappa2 = self.cbf.kappa if kappa is None else jnp.moveaxis(kappa, -1, 0)
            b = h_ddot_drift+(kappa1+kappa2)*h_dot+kappa1*kappa2*h
            domain = jnp.stack([h, h_dot+kappa1*h], axis=1)
        elif self.barrier == "parabolic":
            h, q, A, h_dot_drift = self.parabolic_barrier(state, obstacles)
            b = h_dot_drift+self.cbf.kappa[0]*h
            domain = jnp.stack([h, q], axis=1)
        else:
            raise ValueError("humanoid barrier must be distance or parabolic")
        obstacle_rows = masked_rows(A, b, domain, obstacles.mask)
        return join_constraints(obstacle_rows, self.physical_constraints(state, dt))

    def nominal_control(self, state, goal):
        delta = goal-state[:2]
        desired = self.rotation(state).T @ goal_velocity(
            state[:2], goal, self.position_gain, self.v_max, self.goal_tolerance)
        near_goal = jnp.linalg.norm(delta) < self.goal_tolerance
        yaw_error = angle_normalize(jnp.arctan2(delta[1], delta[0])-state[2])
        omega = jnp.where(near_goal, 0., jnp.clip(self.heading_gain*yaw_error,
                                                -self.omega_max, self.omega_max))
        acceleration = self.velocity_gain*(desired-state[3:5])
        acceleration -= omega*jnp.array([-state[4], state[3]])
        return jnp.concatenate([acceleration, jnp.array([omega])])

    def reward_motion(self, state, action):
        """Return body forward speed and yaw rate."""
        return jnp.stack([state[3], action[2]])
