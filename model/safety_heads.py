"""Learned CBF gains, risk budgets and safety distances."""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp


def _learned_alpha(alpha_logits, hparams):
    alpha_min = jnp.asarray(hparams.alpha_min, dtype=alpha_logits.dtype)
    alpha_range = jnp.asarray(
        hparams.alpha_max - hparams.alpha_min, dtype=alpha_logits.dtype
    )
    return alpha_min + alpha_range * jnn.sigmoid(alpha_logits)


def _fixed_safe_distances(current_obs, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:6 + 6*hparams.num_humans].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(hparams.margin_base, dtype=current_obs.dtype)
    return robot_radius + human_radii + margin_base


def _allocate_beta_budget(beta_logits, mask, hparams):
    active = jnp.clip(mask, 0.0, 1.0) > 0.5
    active_float = active.astype(beta_logits.dtype)
    masked_logits = jnp.where(
        active, beta_logits, jnp.asarray(-1e9, dtype=beta_logits.dtype)
    )
    weights = jnn.softmax(masked_logits, axis=-1) * active_float
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)

    valid_count = jnp.sum(active_float, axis=-1, keepdims=True)
    remaining = jnp.maximum(hparams.beta_max - hparams.beta_min * valid_count, 0.0)
    beta = hparams.beta_min + remaining * weights
    return jnp.where(
        active, beta, jnp.asarray(hparams.beta_min, dtype=beta_logits.dtype)
    )


def _uniform_beta_budget(mask, hparams):
    active = jnp.clip(mask, 0.0, 1.0) > 0.5
    dtype = mask.dtype
    active_count = jnp.sum(active.astype(dtype), axis=-1, keepdims=True)
    beta = jnp.asarray(hparams.beta_max, dtype=dtype) / jnp.maximum(active_count, 1.0)
    return jnp.where(
        active, beta, jnp.asarray(hparams.beta_min, dtype=dtype)
    )


def _safe_distances_from_margin_logits(current_obs, margin_logits, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:6 + 6*hparams.num_humans].reshape((batch_size, hparams.num_humans, 6))
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(hparams.margin_base, dtype=current_obs.dtype)
    margin_extra_max = jnp.asarray(hparams.margin_extra_max, dtype=current_obs.dtype)
    learned_margin = margin_base + margin_extra_max * jnn.sigmoid(margin_logits)
    return robot_radius + human_radii + learned_margin
