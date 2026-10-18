"""Timeout configuration and Brax advantage targets shared by both backends."""
import jax
import jax.numpy as jnp
from trainer.jax_ppo_core import flat_time_batch

def timeout_bootstrap_from_config(config):
    """Keep historical behavior when loading configs without the new switch."""
    legacy_default = config.env.get("backend", "jax") == "crax_mjx"
    value = config.trainer.get("timeout_bootstrap", legacy_default)
    if not isinstance(value, bool):
        raise ValueError("trainer.timeout_bootstrap must be true or false")
    return value


def compute_timeout_gae(rewards, values, terminated, truncated, next_values, gamma, lam):
    """Brax targets with final-observation bootstrap and separate episode ends."""
    rewards, values, next_values = (jnp.asarray(value, jnp.float32)
                                    for value in (rewards, values, next_values))
    gamma, lam = jnp.asarray(gamma, jnp.float32), jnp.asarray(lam, jnp.float32)
    terminated = jnp.asarray(terminated, jnp.float32)
    ended = jnp.maximum(terminated, jnp.asarray(truncated, jnp.float32))
    deltas = rewards + gamma * (1. - terminated) * next_values - values

    def backward(acc, inputs):
        delta, done = inputs
        acc = delta + gamma * lam * (1. - done) * acc
        return acc, acc

    _, corrections = jax.lax.scan(backward, jnp.zeros_like(values[0]), (deltas, ended), reverse=True)
    targets = values + corrections
    shifted = jnp.concatenate([targets[1:], next_values[-1:]], axis=0)
    next_targets = jnp.where(jnp.asarray(truncated, bool), next_values, shifted)
    advantages = rewards + gamma * (1. - terminated) * next_targets - values
    return (jax.lax.stop_gradient(flat_time_batch(advantages)),
            jax.lax.stop_gradient(flat_time_batch(advantages + values)))
