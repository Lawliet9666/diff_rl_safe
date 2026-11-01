"""GNN BarrierNet policy with learned nominal inputs, CBF gains and distances."""

from __future__ import annotations

from model.observation import _body_to_world, _gather_topk
from model.observation import build_qp_context
from model.safety_heads import (
    _fixed_safe_distances,
    _learned_alpha,
    _safe_distances_from_margin_logits,
)

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from model.actor_critic import normalize_robot_type

from model import actor_critic
from model import cbf_qp
from model.gnn_encoder import (
    critic_value_from_latent as shared_critic_value_from_latent,
    current_obs_history,
    init_shared_gnn_actor,
    init_shared_gnn_critic,
    shared_critic_value,
    spatio_temporal_graph_encode,
)
from model.actor_critic import (
    activation,
    cfg_get,
    dense,
    init_linear,
    mlp,
    require_config_value,
)


@dataclass(frozen=True)
class BarrierNetGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    margin_base: float
    margin_extra_max: float
    learn_r_safe: bool
    alpha_min: float
    alpha_max: float
    alpha_init: float
    lookahead_distance: float
    qp_max_iter: int
    qp_eps: float
    qp_not_improved_lim: int
    qp_verbose: int
    qp_check_q_spd: bool
    qp_slack_weight: float
    qp_residual_reward_weight: float
    qp_full_diagnostics: bool
    history_len: int
    obs_frame_dim: int
    num_humans: int
    graph_hidden_dim: int
    graph_latent_dim: int
    graph_layers: int
    position_scale: float
    velocity_scale: float
    qp_obs_top_k: int
    robot: object = None
    robot_state_dim: int = 0
    cbf_order: int = 1
    control_dt: float = 0.1


def hparams_from_config(model_cfg, critic_cfg, act_dim=2, *, robot=None, control_dt=0.1):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "barriernet_gnn", "model")
    require_config_value(
        actor_cfg, "graph_coordinate_frame", "robot_body", "model.actor"
    )
    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    num_humans = int(model_cfg.num_humans)
    qp_obs_top_k = min(
        int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans
    )
    if history_len <= 0:
        raise ValueError("barriernet_gnn requires history_len > 0")
    robot_type = str(model_cfg.robot_type)
    extended = robot_type in {"double_integrator", "humanoid", "quadrotor"}
    if not extended:
        normalize_robot_type(robot_type)
    if extended and robot_type == "humanoid" and robot is not None and robot.barrier not in {"distance", "clearance"}:
        raise ValueError("Humanoid RL requires the distance or clearance HOCBF")
    if extended and float(actor_cfg.qp_slack_weight) <= 0:
        raise ValueError("High-order BarrierNet requires a positive obstacle slack weight")
    if robot is None or int(act_dim) != robot.control_dim:
        raise ValueError("BarrierNet-GNN requires its configured Robot and matching action dimension")
    robot_state_dim = robot.state_dim if extended else 0
    if obs_frame_dim != 6 + 6*int(model_cfg.num_humans) + robot_state_dim:
        raise ValueError("Observation frame must contain navigation, explicit human slots, and full robot state")
    if qp_obs_top_k <= 0:
        raise ValueError("barriernet_gnn requires qp_obs_top_k > 0")

    margin_base = float(actor_cfg.margin_base)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")

    learn_r_safe = bool(cfg_get(actor_cfg, "learn_r_safe", False))
    margin_extra_max = float(cfg_get(actor_cfg, "margin_extra_max", 0.0))
    if not math.isfinite(margin_extra_max) or margin_extra_max < 0.0:
        raise ValueError("model.actor.margin_extra_max must be finite and >= 0")
    if learn_r_safe and margin_extra_max <= 0.0:
        raise ValueError("learn_r_safe requires model.actor.margin_extra_max > 0")

    alpha_min = float(actor_cfg.alpha_min)
    alpha_max = float(actor_cfg.alpha_max)
    alpha_init = float(actor_cfg.alpha_init)
    if not (0.0 < alpha_min < alpha_init < alpha_max):
        raise ValueError(
            "barriernet_gnn requires 0 < alpha_min < alpha_init < alpha_max"
        )

    return BarrierNetGNNHParams(
        act_dim=int(act_dim),
        robot_type=robot_type,
        robot=robot,
        robot_state_dim=robot_state_dim,
        control_dt=float(control_dt),
        cbf_order=(3 if robot_type == "quadrotor" else 2) if extended else 1,
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        margin_base=margin_base,
        margin_extra_max=margin_extra_max,
        learn_r_safe=learn_r_safe,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_init=alpha_init,
        lookahead_distance=float(
            cfg_get(actor_cfg, "lookahead_distance", 0.2)
        ),
        qp_max_iter=int(cfg_get(actor_cfg, "qp_max_iter", 40)),
        qp_eps=float(cfg_get(actor_cfg, "qp_eps", 1e-12)),
        qp_not_improved_lim=int(
            cfg_get(actor_cfg, "qp_not_improved_lim", 3)
        ),
        qp_verbose=int(cfg_get(actor_cfg, "qp_verbose", -1)),
        qp_check_q_spd=bool(cfg_get(actor_cfg, "qp_check_q_spd", False)),
        qp_slack_weight=float(cfg_get(actor_cfg, "qp_slack_weight", 10.0)),
        qp_residual_reward_weight=float(actor_cfg.qp_residual_reward_weight),
        qp_full_diagnostics=bool(
            cfg_get(actor_cfg, "qp_full_diagnostics", False)
        ),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        num_humans=num_humans,
        graph_hidden_dim=int(cfg_get(actor_cfg, "graph_hidden_dim", 64)),
        graph_latent_dim=int(cfg_get(actor_cfg, "graph_latent_dim", 16)),
        graph_layers=int(cfg_get(actor_cfg, "graph_layers", 3)),
        position_scale=float(cfg_get(actor_cfg, "position_scale", 20.0)),
        velocity_scale=float(cfg_get(actor_cfg, "velocity_scale", 1.0)),
        qp_obs_top_k=qp_obs_top_k,
    )


def init_params_from_config(
    key,
    model_cfg,
    critic_cfg,
    act_dim,
    action_low,
    action_high,
    use_init_weights=True,
    *, robot=None, control_dt=0.1,
):
    actor_cfg = model_cfg.actor
    hparams = hparams_from_config(model_cfg, critic_cfg, act_dim=act_dim, robot=robot, control_dt=control_dt)
    hidden_dim = int(cfg_get(actor_cfg, "hidden_dim", 256))
    control_hidden_dim = int(
        cfg_get(actor_cfg, "control_hidden_dim", hidden_dim)
    )
    scalar_hidden_dim = int(
        cfg_get(actor_cfg, "scalar_hidden_dim", hidden_dim)
    )
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    sqrt2 = math.sqrt(2.0)

    keys = jax.random.split(key, 12)
    actor = init_shared_gnn_actor(
        keys[:4],
        hparams,
        act_dim,
        hidden_dim,
        control_hidden_dim,
        actor_cfg.action_std_init,
        "nominal_head",
        use_init_weights,
    )
    actor["human_fc1"] = init_linear(
        keys[4],
        hparams.graph_latent_dim,
        scalar_hidden_dim,
        sqrt2,
        use_init_weights,
    )
    actor["human_alpha_head"] = init_linear(
        keys[5], scalar_hidden_dim, hparams.cbf_order, sqrt2 * 0.01, use_init_weights
    )
    alpha_fraction = (hparams.alpha_init - hparams.alpha_min) / (
        hparams.alpha_max - hparams.alpha_min
    )
    alpha_bias = math.log(alpha_fraction / (1.0 - alpha_fraction))
    actor["human_alpha_head"]["bias"] = jnp.full(
        (hparams.cbf_order,), alpha_bias, dtype=jnp.float32
    )
    if hparams.learn_r_safe:
        actor["human_rsafe_head"] = init_linear(
            keys[6], scalar_hidden_dim, 1, sqrt2 * 0.01, use_init_weights
        )
    critic = init_shared_gnn_critic(
        keys[7:10],
        hparams,
        critic_hidden_dim,
        critic_hidden_dim2,
        use_init_weights,
    )
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _current_obs(obs, hparams):
    obs_history = current_obs_history(obs, hparams)
    return obs_history, obs_history[:, -1, :]


def _actor_outputs(params, hparams, obs, qp_context=None):
    obs_history, current_obs = _current_obs(obs, hparams)
    actor_params = params["actor"]
    robot_latent, human_latents = spatio_temporal_graph_encode(
        actor_params["graph_encoder"],
        hparams,
        obs_history,
        return_human_latents=True,
    )
    robot_x = activation(
        dense(actor_params["fc1"], robot_latent), hparams.actor_act
    )
    nominal_xy = mlp(
        actor_params["nominal_head"], robot_x, hparams.actor_act
    )

    human_x = activation(
        dense(actor_params["human_fc1"], human_latents), hparams.actor_act
    )
    alpha_logits = dense(actor_params["human_alpha_head"], human_x)
    if hparams.cbf_order == 1:
        alpha_logits = alpha_logits.squeeze(-1)
    alpha_all = _learned_alpha(alpha_logits, hparams)
    if hparams.learn_r_safe:
        margin_logits = dense(actor_params["human_rsafe_head"], human_x).squeeze(-1)
        r_safe_all = _safe_distances_from_margin_logits(
            current_obs, margin_logits, hparams
        )
    else:
        r_safe_all = _fixed_safe_distances(current_obs, hparams)

    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    qp_topk_ids = qp_context["qp_topk_ids"]
    alpha_topk = _gather_topk(alpha_all, qp_topk_ids)
    r_safe_topk = _gather_topk(r_safe_all, qp_topk_ids)
    if hparams.robot_state_dim == 0:
        nominal_xy = _body_to_world(nominal_xy, qp_obs[:, 4])
    safe_xy, slack = cbf_qp.solve_slack_qp(
        params,
        hparams,
        qp_obs,
        nominal_xy,
        alpha_topk,
        r_safe_topk,
        robot_state=qp_context.get("robot_state"),
    )
    return (
        robot_latent,
        qp_obs,
        nominal_xy,
        alpha_topk,
        r_safe_topk,
        safe_xy,
        slack,
    )


def safe_action_mean(params, hparams, obs, qp_context=None):
    outputs = _actor_outputs(params, hparams, obs, qp_context=qp_context)
    return outputs[-2]


policy_action_mean = safe_action_mean
policy_action_to_env_action = actor_critic.policy_action_to_env_action


def critic_value(params, hparams, obs):
    return shared_critic_value(params, hparams, obs)


def get_action_and_value(
    params,
    hparams,
    obs,
    action=None,
    noise=None,
    policy_action=None,
    return_policy_action=False,
    return_diagnostics=False,
    qp_context=None,
):
    (
        robot_latent,
        qp_obs,
        nominal_xy,
        alpha,
        r_safe,
        safe_xy,
        slack,
    ) = _actor_outputs(params, hparams, obs, qp_context=qp_context)

    action, logprob, entropy, sampled_policy_action = (
        actor_critic.action_logprob_entropy(
            params,
            safe_xy,
            action=action,
            noise=noise,
            policy_action=policy_action,
        )
    )
    value = shared_critic_value_from_latent(
        params, hparams, robot_latent
    )
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        if hparams.qp_full_diagnostics:
            diagnostics = cbf_qp.full_diagnostics(
                hparams,
                qp_obs,
                nominal_xy,
                alpha,
                r_safe,
                safe_xy,
                slack,
                robot_state=None if qp_context is None else qp_context.get("robot_state"),
            )
        else:
            diagnostics = cbf_qp.lightweight_diagnostics(
                hparams,
                qp_obs,
                nominal_xy,
                alpha,
                r_safe,
                safe_xy,
                slack,
                robot_state=None if qp_context is None else qp_context.get("robot_state"),
            )
        outputs = outputs + (diagnostics,)
    return outputs
