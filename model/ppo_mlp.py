"""Pure PPO actor/critic with a shared MLP encoder and no graph network."""

from __future__ import annotations

from model.mlp_encoder import _critic_value_from_latent, _init_encoder, _shared_latent, init_shared_mlp_critic

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from model.actor_critic import normalize_robot_type
from model import actor_critic
from model.actor_critic import (
    activation,
    cfg_get,
    dense,
    init_head,
    init_linear,
    mlp,
    require_config_value,
)


@dataclass(frozen=True)
class PPOMLPHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    history_len: int
    obs_frame_dim: int
    obs_dim: int
    num_humans: int
    encoder_hidden_dim: int
    encoder_latent_dim: int
    encoder_layers: int
    position_scale: float
    velocity_scale: float

    @property
    def encoder_input_dim(self):
        processed_frame_dim = 5 + 6 * int(self.num_humans)
        return int(self.history_len) * processed_frame_dim


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "ppo_mlp", "model")
    require_config_value(actor_cfg, "coordinate_frame", "robot_body", "model.actor")

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("ppo_mlp requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("ppo_mlp requires obs_frame_dim = 6 + num_humans * 6")
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError("ppo_mlp requires obs_dim = history_len * obs_frame_dim")
    if int(act_dim) != 2:
        raise ValueError("ppo_mlp requires act_dim=2")

    encoder_hidden_dim = int(cfg_get(actor_cfg, "encoder_hidden_dim", 64))
    encoder_latent_dim = int(cfg_get(actor_cfg, "encoder_latent_dim", 16))
    encoder_layers = int(cfg_get(actor_cfg, "encoder_layers", 3))
    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if encoder_hidden_dim <= 0 or encoder_latent_dim <= 0 or encoder_layers <= 0:
        raise ValueError("ppo_mlp encoder dimensions and layer count must be > 0")
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError("ppo_mlp position_scale and velocity_scale must be > 0")

    return PPOMLPHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=(obs_frame_dim - 6) // 6,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_latent_dim=encoder_latent_dim,
        encoder_layers=encoder_layers,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )


def init_params_from_config(
    key,
    model_cfg,
    critic_cfg,
    act_dim,
    action_low,
    action_high,
    use_init_weights=True,
):
    actor_cfg = model_cfg.actor
    hparams = hparams_from_config(model_cfg, critic_cfg, act_dim=act_dim)
    hidden_dim = int(cfg_get(actor_cfg, "hidden_dim", 256))
    control_hidden_dim = int(cfg_get(actor_cfg, "control_hidden_dim", hidden_dim))
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(hidden_dim, control_hidden_dim, critic_hidden_dim, critic_hidden_dim2) <= 0:
        raise ValueError("ppo_mlp actor and critic hidden dimensions must be > 0")

    keys = jax.random.split(key, hparams.encoder_layers + 6)
    encoder_end = hparams.encoder_layers
    sqrt2 = math.sqrt(2.0)
    actor = {
        "logstd": jnp.full((act_dim,), math.log(float(actor_cfg.action_std_init)), dtype=jnp.float32),
        "mlp_encoder": _init_encoder(keys[:encoder_end], hparams, use_init_weights),
        "fc1": init_linear(
            keys[encoder_end],
            hparams.encoder_latent_dim,
            hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "control_head": init_head(
            keys[encoder_end + 1 : encoder_end + 3],
            hidden_dim,
            control_hidden_dim,
            act_dim,
            use_init_weights,
        ),
    }
    critic = init_shared_mlp_critic(keys[encoder_end + 3:encoder_end + 6],
                                    hparams, critic_hidden_dim, critic_hidden_dim2, use_init_weights)
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _actor_mean_from_latent(params, hparams, latent):
    hidden = activation(dense(params["actor"]["fc1"], latent), hparams.actor_act)
    return mlp(params["actor"]["control_head"], hidden, hparams.actor_act)


def policy_action_mean(params, hparams, obs):
    """Return the deterministic pre-tanh action mean."""
    latent = _shared_latent(params, hparams, obs)
    return _actor_mean_from_latent(params, hparams, latent)


policy_action_to_env_action = actor_critic.policy_action_to_env_action


def critic_value(params, hparams, obs):
    latent = _shared_latent(params, hparams, obs)
    return _critic_value_from_latent(params, hparams, latent)


def get_action_and_value(
    params,
    hparams,
    obs,
    action=None,
    noise=None,
    policy_action=None,
    return_policy_action=False,
):
    latent = _shared_latent(params, hparams, obs)
    mean = _actor_mean_from_latent(params, hparams, latent)
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        mean,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )
    value = _critic_value_from_latent(params, hparams, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    return outputs
