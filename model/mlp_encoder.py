"""Shared MLP observation encoder and critic helpers."""

from __future__ import annotations

import math
import jax.numpy as jnp
from model.observation import _world_to_body
from model.ppo_base import activation, dense, init_linear, mlp


def _init_encoder(keys, hparams, use_init_weights):
    dims = (
        [hparams.encoder_input_dim]
        + [hparams.encoder_hidden_dim]
        * max(hparams.encoder_layers - 1, 0)
        + [hparams.encoder_latent_dim]
    )
    sqrt2 = math.sqrt(2.0)
    return [
        init_linear(key, in_dim, out_dim, sqrt2, use_init_weights)
        for key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:])
    ]


def _obs_history(obs, hparams):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim != 2:
        raise ValueError(
            f"MLP policy expects rank-2 observations, got ndim={obs.ndim}"
        )
    if obs.shape[-1] != hparams.obs_dim:
        raise ValueError(
            f"MLP policy expects obs_dim={hparams.obs_dim}, got {obs.shape[-1]}"
        )
    return obs.reshape(
        (obs.shape[0], hparams.history_len, hparams.obs_frame_dim)
    )


def _preprocess_obs(obs, hparams):
    frames = _obs_history(obs, hparams)
    robot_theta = frames[:, :, 4]
    goal_rel = (
        _world_to_body(frames[:, :, 0:2], robot_theta)
        / hparams.position_scale
    )
    robot_vel = (
        _world_to_body(frames[:, :, 2:4], robot_theta)
        / hparams.velocity_scale
    )
    robot_radius = frames[:, :, 5:6] / hparams.position_scale

    human_blocks = frames[:, :, 6:].reshape(
        (frames.shape[0], hparams.history_len, hparams.num_humans, 6)
    )
    human_mask = jnp.clip(human_blocks[:, :, :, 5:6], 0.0, 1.0)
    human_rel = (
        _world_to_body(human_blocks[:, :, :, 0:2], robot_theta)
        / hparams.position_scale
    )
    human_vel = (
        _world_to_body(human_blocks[:, :, :, 2:4], robot_theta)
        / hparams.velocity_scale
    )
    human_radius = human_blocks[:, :, :, 4:5] / hparams.position_scale
    human_values = jnp.concatenate(
        [human_rel, human_vel, human_radius], axis=-1
    ) * human_mask
    human_features = jnp.concatenate(
        [human_values, human_mask], axis=-1
    )

    processed_frames = jnp.concatenate(
        [
            goal_rel,
            robot_vel,
            robot_radius,
            human_features.reshape(
                (frames.shape[0], hparams.history_len, -1)
            ),
        ],
        axis=-1,
    )
    return processed_frames.reshape((frames.shape[0], -1))


def _shared_latent(params, hparams, obs):
    features = _preprocess_obs(obs, hparams)
    return mlp(
        params["actor"]["mlp_encoder"], features, hparams.actor_act
    )


def _critic_value_from_latent(params, hparams, latent):
    hidden = activation(
        dense(params["critic"]["fc1"], latent), hparams.critic_act
    )
    hidden = activation(
        dense(params["critic"]["fc21"], hidden), hparams.critic_act
    )
    return dense(params["critic"]["fc31"], hidden).squeeze(-1)


def init_shared_mlp_critic(keys, hparams, hidden_dim, hidden_dim2, use_init_weights):
    sqrt2 = math.sqrt(2.0)
    return {
        "fc1": init_linear(keys[0], hparams.encoder_latent_dim, hidden_dim, sqrt2, use_init_weights),
        "fc21": init_linear(keys[1], hidden_dim, hidden_dim2, sqrt2, use_init_weights),
        "fc31": init_linear(keys[2], hidden_dim2, 1, 1.0, use_init_weights),
    }
