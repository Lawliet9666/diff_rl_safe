"""Mutually exclusive evaluation outcomes for safe goal reaching."""

import jax.numpy as jnp


def episode_outcomes(done, success, collision):
    """Return boolean (success, collision, timeout), with collision priority."""
    ended = jnp.asarray(done) > 0.5
    collision = ended & (jnp.asarray(collision) > 0.5)
    success = ended & (jnp.asarray(success) > 0.5) & ~collision
    timeout = ended & ~(success | collision)
    return success, collision, timeout
