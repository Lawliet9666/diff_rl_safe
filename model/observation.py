"""Policy observation geometry and nearest-obstacle selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _world_to_body(vectors, robot_theta):
    c = jnp.cos(robot_theta)
    s = jnp.sin(robot_theta)
    while c.ndim < vectors.ndim - 1:
        c = c[..., None]
        s = s[..., None]
    x = vectors[..., 0]
    y = vectors[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)


def _body_to_world(vectors, theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return jnp.stack([c * x - s * y, s * x + c * y], axis=1)


def _select_qp_topk(current_obs, human_ids, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:6 + 6*hparams.num_humans].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    mask = jnp.clip(blocks[:, :, 5], 0.0, 1.0) > 0.5
    distance = jnp.linalg.norm(blocks[:, :, 0:2], axis=2)
    sort_key = jnp.where(mask, distance, jnp.inf)
    _values, topk_ids = jax.lax.top_k(
        -sort_key, hparams.qp_obs_top_k
    )
    batch_idx = jnp.arange(batch_size)[:, None]
    topk_blocks = blocks[batch_idx, topk_ids]
    selected_human_ids = human_ids[batch_idx, topk_ids]
    qp_obs = jnp.concatenate(
        [current_obs[:, :6], topk_blocks.reshape((batch_size, -1))], axis=1
    )
    return qp_obs, selected_human_ids


def _gather_topk(values, qp_topk_ids):
    batch_idx = jnp.arange(values.shape[0])[:, None]
    return values[batch_idx, qp_topk_ids.astype(jnp.int32)]


def build_qp_context(obs, hparams, *, robot_state=None):
    """Select nearest obstacles and retain the supplied robot-state precision."""
    frames = jnp.asarray(obs, dtype=jnp.float32).reshape(
        (obs.shape[0], hparams.history_len, hparams.obs_frame_dim)
    )
    current_obs = frames[:, -1, :]
    human_ids = jnp.broadcast_to(
        jnp.arange(hparams.num_humans, dtype=jnp.int32)[None, :],
        (current_obs.shape[0], hparams.num_humans),
    )
    qp_obs, ids = _select_qp_topk(current_obs, human_ids, hparams)
    context = {"qp_obs": qp_obs, "qp_topk_ids": ids.astype(jnp.int32)}
    if getattr(hparams, "robot_state_dim", 0) and robot_state is not None:
        if robot_state.shape != (obs.shape[0], hparams.robot_state_dim):
            raise ValueError("robot_state shape does not match the configured Robot")
        context["robot_state"] = robot_state  # Preserve dynamics precision.
    return context


def robot_features(frames, hparams):
    """Robot-state features excluding absolute x/y, already encoded by navigation."""
    start = 6 + 6*hparams.num_humans
    return frames[..., start+2:] if hparams.robot_state_dim else frames[..., :0]
