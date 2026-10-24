"""Shared robot configuration, JAX constraint tools and stateless robot contract."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp


class RobotStep(NamedTuple):
    state: Array
    velocity: Array
    action: Array


def angle_normalize(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


class Obstacles(NamedTuple):
    """Single scene: centers/velocities [N,2], radii/mask [N].

    Velocities are world-frame and held constant in the barrier prediction.
    Masked slots may contain arbitrary finite values; they add no constraint.
    """
    centers: Array
    velocities: Array
    radii: Array
    mask: Array


class Constraints(NamedTuple):
    """A u + b >= 0; domain contains h and recursive HOCBF initial values."""
    A: Array
    b: Array
    domain: Array


def check_obstacle_shapes(obstacles):
    n = obstacles.centers.shape[0]
    if (obstacles.centers.shape != (n, 2)
            or obstacles.velocities.shape != (n, 2)
            or obstacles.radii.shape != (n,) or obstacles.mask.shape != (n,)):
        raise ValueError("expected centers/velocities [N,2] and radii/mask [N]")


def masked_rows(A, b, domain, mask):
    active = mask > 0.5
    # A strictly feasible 0*u+1>=0 row avoids a degenerate masked equality.
    return Constraints(jnp.where(active[:, None], A, 0.0),
                       jnp.where(active, b, 1.0),
                       jnp.where(active[:, None], domain, 1.0).reshape(-1))


def join_constraints(*blocks):
    return Constraints(*(jnp.concatenate([b[i] for b in blocks], axis=0)
                         for i in range(3)))


def _hocbf(h, h_dot, h_ddot, dh_ddot_dx_g, h_dddot_drift, kappa):
    # Recursive psi_i = d(psi_{i-1})/dt + kappa_i*psi_{i-1}; psi_0 = h.
    kappa1, kappa2, kappa3 = kappa
    psi1 = h_dot+kappa1*h
    psi2 = h_ddot+(kappa1+kappa2)*h_dot+kappa1*kappa2*h
    b = (h_dddot_drift+(kappa1+kappa2+kappa3)*h_ddot
         +(kappa1*kappa2+kappa1*kappa3+kappa2*kappa3)*h_dot+kappa1*kappa2*kappa3*h)
    domain = jnp.stack([h, psi1, psi2], axis=-1).reshape(-1)
    return Constraints(dh_ddot_dx_g, b, domain)


def _control_parameters(section, names, label, *, selected_cbf=None):
    """Flatten loaded controller parameters, visiting only the selected CBF group."""
    groups = ("nominal", "cbf", "qp")
    if not isinstance(section, Mapping):
        raise ValueError(f"{label} must be a mapping")
    missing = set(groups)-section.keys()
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
    parameters = {}

    def collect(node):
        for name, value in node.items():
            if isinstance(value, Mapping):
                collect(value)
            else:
                if name in parameters:
                    raise ValueError(f"{label} has duplicate parameters: {name}")
                parameters[name] = value

    for group in groups:
        if not isinstance(section[group], Mapping):
            raise ValueError(f"{label} parameter groups must be mappings")
    if selected_cbf is not None:
        if selected_cbf not in section["cbf"]:
            raise ValueError(f"{label}.cbf is missing required fields: {selected_cbf}")
        active = section["cbf"][selected_cbf]
        if not isinstance(active, Mapping):
            raise ValueError(f"{label}.cbf.{selected_cbf} must be a mapping")
        section = {**section, "cbf": active}
    collect(section)
    missing = set(names)-parameters.keys()
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
    extra = parameters.keys()-set(names)
    if extra:
        raise ValueError(f"{label} has unexpected parameters: {', '.join(sorted(extra))}")
    return parameters


class Robot(ABC):
    state_dim: int
    control_dim: int

    @abstractmethod
    def dynamics(self, state, action):
        """Continuous-time derivative for one robot state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state, action, dt):
        """Return next state, world velocity and applied physical input."""
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, pose):
        """Create the complete model state from a sampled [x, y, yaw] pose."""
        raise NotImplementedError

    def check_state_limits(self, state, tolerance=1e-6):
        """Return whether the state is finite and satisfies physical state limits."""
        return jnp.all(jnp.isfinite(state))

    @abstractmethod
    def input_bounds(self):
        """Componentwise physical input bounds, as (lower, upper)."""
        raise NotImplementedError

    @abstractmethod
    def planar_pose(self, state):
        """Return the [x, y, yaw] projection used by the crowd and renderer."""
        raise NotImplementedError

    def cbf_constraints(self, state, obstacles, dt):
        """Return A, b and initial domain conditions for A@u+b >= 0."""
        raise NotImplementedError(
            f"CBF constraints are not implemented for {type(self).__name__}")

    def nominal_control(self, state, goal):
        """Return a goal-tracking reference without applying a safety filter."""
        raise NotImplementedError(
            f"Nominal control is not implemented for {type(self).__name__}")

    def control_output_matrix(self, state):
        """Map physical control to the motion output tracked by the controller."""
        raise NotImplementedError(
            f"Control output mapping is not implemented for {type(self).__name__}")
