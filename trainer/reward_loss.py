"""Pathwise return objective for windows of the PPO rollout."""

import math

import jax.numpy as jnp


def discounted_return(batch, gamma):
    """Use the first episode segment; bootstrap timeouts before auto-reset."""
    rewards = batch["rew"]
    done = batch["done"] > 0.5
    alive = jnp.concatenate([
        jnp.ones_like(done[:1]),
        jnp.cumprod(~done[:-1], axis=0).astype(bool),
    ], axis=0)
    discounts = jnp.asarray(gamma, rewards.dtype) ** jnp.arange(rewards.shape[0])
    returns = jnp.sum(discounts[:, None] * jnp.where(alive, rewards, 0.0), axis=0)
    timeout_value = jnp.where(alive & (batch["truncated"] > 0.5),
                              batch["next_values"], 0.0)
    returns += jnp.sum(discounts[:, None] * gamma * timeout_value, axis=0)
    returns += gamma ** rewards.shape[0] * jnp.where(
        jnp.all(~done, axis=0), batch["next_value"], 0.0)
    return returns


def make_reward_loss(config):
    section = config.trainer.reward_loss
    weight = float(section.weight)
    if isinstance(section.weight, bool) or not math.isfinite(weight) or weight < 0:
        raise ValueError("trainer.reward_loss.weight must be finite and >= 0")
    if weight == 0:
        return None
    if config.env.get("backend", "jax") != "jax" or config.model.type not in {
            "barriernet_gnn", "barriernet_mlp"}:
        raise ValueError("reward_loss requires BarrierNet and the JAX navigation environment")
    if config.trainer.safety_filter.enabled:
        raise ValueError("reward_loss cannot use the external PPO safety filter")
    horizon = section.horizon
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("trainer.reward_loss.horizon must be a positive integer")
    gamma = float(config.trainer.gamma)

    def reward_loss(transitions, next_value):
        raw_loss = -jnp.mean(discounted_return({**transitions, "next_value": next_value}, gamma))
        return weight * raw_loss, raw_loss

    return reward_loss
