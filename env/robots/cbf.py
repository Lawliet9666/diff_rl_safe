"""Pure JAX obstacle and barrier data; no controller or QP solver dependencies."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Obstacles(NamedTuple):
    """Single scene: centers/velocities [N,2], radii/mask [N].

    Velocities are world-frame and held constant in the barrier prediction.
    Masked slots may contain arbitrary finite values; they add no constraint.
    """
    centers: jax.Array
    velocities: jax.Array
    radii: jax.Array
    mask: jax.Array


class Constraints(NamedTuple):
    """A u + b >= 0; domain contains h and recursive HOCBF initial values."""
    A: jax.Array
    b: jax.Array
    domain: jax.Array


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
