"""Fixed-ID JAX rollout collection for supported PPO policy modules."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from trainer.ppo_core import flat_time_batch, _trainable_params, _with_trainable
from env.robots.base import Obstacles


def _topk_mask(values, qp_obs):
    human_blocks = qp_obs[..., 6:].reshape((*qp_obs.shape[:-1], -1, 6))
    mask = jnp.clip(human_blocks[..., 5], 0.0, 1.0).astype(values.dtype)
    if values.ndim > mask.ndim:
        mask = jnp.broadcast_to(mask[..., None], values.shape)
    return mask


def _masked_topk_sum_count(values, qp_obs):
    mask = _topk_mask(values, qp_obs)
    return jnp.sum(values * mask), jnp.sum(mask, dtype=jnp.float32)


def _masked_topk_stats(values, qp_obs):
    mask = _topk_mask(values, qp_obs)
    active = mask > 0.0
    count = jnp.sum(mask, dtype=jnp.float32)
    total = jnp.sum(values * mask)
    denominator = jnp.maximum(count, 1.0)
    mean = total / denominator
    variance = jnp.sum(jnp.square(values - mean) * mask) / denominator
    minimum = jnp.where(
        count > 0.0, jnp.min(jnp.where(active, values, jnp.inf)), 0.0
    )
    maximum = jnp.where(
        count > 0.0, jnp.max(jnp.where(active, values, -jnp.inf)), 0.0
    )
    return minimum, maximum, jnp.sqrt(jnp.maximum(variance, 0.0))


def _beta_budget_error_max(beta, qp_obs, beta_budget):
    mask = _topk_mask(beta, qp_obs)
    beta_totals = jnp.sum(beta * mask, axis=-1)
    target = jnp.where(jnp.sum(mask, axis=-1) > 0.0, beta_budget, 0.0)
    return jnp.max(jnp.abs(beta_totals - target))


def _rollout_summary(transitions, has_policy_qp, beta_budget=None):
    completed = jnp.isfinite(transitions["completed_return"])
    completed_returns = jnp.where(completed, transitions["completed_return"], 0.0)
    completed_ppo_returns = jnp.where(completed, transitions["completed_ppo_return"], 0.0)
    completed_lengths = jnp.where(completed, transitions["completed_length"], 0.0)
    summary = {
        "train_completed_count": completed.astype(jnp.int32).sum(),
        "train_return_sum": completed_returns.sum(),
        "train_ppo_return_sum": completed_ppo_returns.sum(),
        "train_length_sum": completed_lengths.sum(),
        "train_success_count": transitions["success"].astype(jnp.int32).sum(),
        "train_collision_count": transitions["collision"].astype(jnp.int32).sum(),
        "train_timeout_count": transitions["timeout"].astype(jnp.int32).sum(),
        "jax_env_min_clearance": jnp.nanmin(transitions["min_clearance"]),
        "boundary_violation_mean": jnp.mean(transitions["boundary_violation"]),
        "boundary_violation_max": jnp.max(transitions["boundary_violation"]),
        "robot_omega_delta_mean": jnp.mean(transitions["robot_omega_delta"]),
        "robot_omega_delta_max": jnp.max(transitions["robot_omega_delta"]),
        "robot_omega_delta_penalty_mean": jnp.mean(transitions["robot_omega_delta_penalty"]),
    }
    if has_policy_qp:
        r_safe_sum, r_safe_count = _masked_topk_sum_count(
            transitions["r_safe"], transitions["qp_obs"]
        )
        summary.update({
            "qp_slack_sum": transitions["qp_slack"].sum(),
            "qp_slack_count": jnp.asarray(transitions["qp_slack"].size, dtype=jnp.float32),
            "qp_slack_max": jnp.max(transitions["qp_slack_max"]),
            "qp_correction_sum": transitions["qp_correction"].sum(),
            "qp_correction_count": jnp.asarray(transitions["qp_correction"].size, dtype=jnp.float32),
            "r_safe_sum": r_safe_sum,
            "r_safe_count": r_safe_count,
        })
        if "beta" in transitions:
            beta_active_min, beta_active_max, beta_active_std = (
                _masked_topk_stats(
                    transitions["beta"], transitions["qp_obs"]
                )
            )
            summary.update({
                "beta_active_min": beta_active_min,
                "beta_active_max": beta_active_max,
                "beta_active_std": beta_active_std,
                "beta_budget_error_max": _beta_budget_error_max(
                    transitions["beta"], transitions["qp_obs"], beta_budget
                ),
            })
        if "alpha" in transitions:
            alpha_active_min, alpha_active_max, alpha_active_std = (
                _masked_topk_stats(
                    transitions["alpha"], transitions["qp_obs"]
                )
            )
            summary.update({
                "alpha_active_min": alpha_active_min,
                "alpha_active_max": alpha_active_max,
                "alpha_active_std": alpha_active_std,
            })
        if "nominal_cbf_residual" in transitions:
            summary.update({
                "nominal_cbf_residual_sum": transitions["nominal_cbf_residual"].sum(),
                "nominal_cbf_residual_count": jnp.asarray(
                    transitions["nominal_cbf_residual"].size, dtype=jnp.float32
                ),
                "nominal_cbf_residual_max": jnp.max(transitions["nominal_cbf_residual_max"]),
                "safe_cbf_residual_sum": transitions["safe_cbf_residual"].sum(),
                "safe_cbf_residual_count": jnp.asarray(
                    transitions["safe_cbf_residual"].size, dtype=jnp.float32
                ),
                "safe_cbf_residual_max": jnp.max(transitions["safe_cbf_residual_max"]),
            })
    return summary


def _select_done(done_bool, reset_value, next_value):
    if next_value.ndim == 0:
        return next_value
    shape = (done_bool.shape[0],) + (1,) * (next_value.ndim - 1)
    return jnp.where(done_bool.reshape(shape), reset_value, next_value)


def _maybe_reset_done_envs(env_cfg, next_state, reset_key, done_bool, batch_size, reset_fn):
    def reset_done(_unused):
        reset_state = reset_fn(env_cfg, reset_key, batch_size)
        return jax.tree_util.tree_map(
            lambda reset_value, next_value: _select_done(done_bool, reset_value, next_value),
            reset_state,
            next_state,
        )

    def keep_next(_unused):
        return next_state

    return jax.lax.cond(jnp.any(done_bool), reset_done, keep_next, operand=None)


def _update_gnn_obs_history(history, next_frame, reset_mask):
    shifted = jnp.concatenate([history[:, 1:, :], next_frame[:, None, :]], axis=1)
    reset_history = jnp.repeat(next_frame[:, None, :], history.shape[1], axis=1)
    return jnp.where(reset_mask[:, None, None], reset_history, shifted)


def _update_ppo_episode_returns(running_returns, reward, done_bool):
    completed_returns = running_returns + reward
    next_returns = jnp.where(done_bool, 0.0, completed_returns)
    logged_returns = jnp.where(done_bool, completed_returns, jnp.nan)
    return next_returns.astype(jnp.float32), logged_returns.astype(jnp.float32)


def _collect_reward_windows(body, initial, params, value_fn, loss_fn, horizon, steps):
    """Differentiate each window once, detaching its start and sharing its transitions."""
    def window(carry, length):
        carry = jax.tree_util.tree_map(jax.lax.stop_gradient, carry)

        def loss(trainable):
            policy_params = _with_trainable(params, trainable)
            final, transitions = jax.lax.scan(
                lambda state, _: body(policy_params, state, None), carry, None, length=length)
            weighted, raw = loss_fn(transitions, value_fn(final[1]))
            return weighted, (raw, final, transitions)

        (weighted, (raw, final, transitions)), grads = jax.value_and_grad(
            loss, has_aux=True)(_trainable_params(params))
        return final, transitions, grads, weighted, raw

    first_length = min(horizon, steps)
    carry, first, grads, weighted, raw = window(initial, first_length)
    pieces = [first]
    remaining = steps - first_length
    full_windows, tail = divmod(remaining, horizon)

    def accumulate(carry, _):
        state, total_grads, total_weighted, total_raw = carry
        state, transitions, grad, loss, raw_loss = window(state, horizon)
        total_grads = jax.tree_util.tree_map(jnp.add, total_grads, grad)
        return (state, total_grads, total_weighted + loss, total_raw + raw_loss), transitions

    if full_windows:
        (carry, grads, weighted, raw), rest = jax.lax.scan(
            accumulate, (carry, grads, weighted, raw), None, length=full_windows)
        pieces.append(jax.tree_util.tree_map(
            lambda value: value.reshape((-1,) + value.shape[2:]), rest))
    if tail:
        carry, last, grad, loss, raw_loss = window(carry, tail)
        grads = jax.tree_util.tree_map(jnp.add, grads, grad)
        weighted, raw = weighted + loss, raw + raw_loss
        pieces.append(last)
    count = 1 + full_windows + bool(tail)
    transitions = jax.tree_util.tree_map(lambda *values: jnp.concatenate(values), *pieces)
    metrics = {
        "reward_gradients": jax.tree_util.tree_map(lambda grad: jax.lax.stop_gradient(grad / count), grads),
        "reward_loss/raw": raw / count,
        "reward_loss/weighted": weighted / count,
        "reward_loss/return": -raw / count,
        "reward_loss/windows": jnp.asarray(count, jnp.float32),
        "reward_loss/shared_steps": jnp.asarray(steps * initial[1].shape[0], jnp.float32),
        "reward_loss/simulated_steps": jnp.asarray(0., jnp.float32),
    }
    return carry, transitions, metrics


def _cbf_reward(proposed, filtered, diagnostics, sigma):
    """CBF-RL shaping using the active obstacle with minimum surface clearance."""
    mask = diagnostics["obstacle_mask"]
    residual = diagnostics["proposed_cbf_residual"]
    clearance = diagnostics["obstacle_clearance"]
    if residual.shape[-1] == 0:
        violation = jnp.zeros(residual.shape[:-1], dtype=residual.dtype)
    else:
        active = mask > 0.5
        index = jnp.argmin(jnp.where(active, clearance, jnp.inf), axis=-1)
        selected = jnp.take_along_axis(residual, index[..., None], axis=-1)[..., 0]
        violation = jnp.where(jnp.any(active, axis=-1), jnp.minimum(selected, 0.), 0.)
    correction = jnp.expm1(-jnp.sum(jnp.square(proposed-filtered), axis=-1) / sigma**2)
    return jax.tree_util.tree_map(lambda value: jax.lax.stop_gradient(value.astype(jnp.float32)), {
        "cbf_rl/violation": violation,
        "cbf_rl/correction": correction,
        "cbf_rl/reward": violation + correction,
    })


def make_rollout_collector(model_module, *, step_fn, reset_fn,
                           observe_fn, timeout_aware=False, safety_filter=None,
                           reward_loss_fn=None, reward_horizon=8,
                           cbf_reward_weight=0.0, cbf_reward_sigma=0.5):
    """Infer policy-QP support once when creating the collector, before JIT tracing."""
    has_policy_qp = hasattr(model_module, "build_qp_context")
    if safety_filter is not None and has_policy_qp:
        raise ValueError("external safety filter cannot wrap a policy with an internal QP")

    if reward_loss_fn is not None and (reward_horizon <= 0 or safety_filter is not None):
        raise ValueError("shared reward loss needs a positive horizon and no external filter")
    for name, value in (("weight", cbf_reward_weight), ("sigma", cbf_reward_sigma)):
        if (isinstance(value, bool) or not math.isfinite(value)
                or value < 0 or (name == "sigma" and value == 0)):
            raise ValueError(f"invalid safety_filter.reward.{name}")
    if cbf_reward_weight > 0 and safety_filter is None:
        raise ValueError("CBF-RL reward requires an external obstacle safety filter")
    need_timeout_values = timeout_aware or reward_loss_fn is not None

    def collect(
        params,
        hparams,
        env_cfg,
        env_state,
        rng,
        steps_per_env,
        global_step=0,
        value_params=None,
    ):
        value_params = params if value_params is None else value_params
        if reward_loss_fn is not None:
            value_params = jax.tree_util.tree_map(jax.lax.stop_gradient, value_params)
        nav_state, obs_history, ppo_episode_returns = env_state
        batch_size = int(obs_history.shape[0])
        act_dim = int(hparams.act_dim)

        def body(policy_params, carry, _unused):
            state, history, running_ppo_returns, key = carry
            key, noise_key, env_key, reset_key = jax.random.split(key, 4)
            policy_obs = history.reshape((batch_size, -1))
            noise = jax.random.normal(noise_key, (batch_size, act_dim), dtype=jnp.float32)
            if has_policy_qp:
                context_kwargs = dict(robot_state=state.robot_state) if getattr(hparams, "robot_state_dim", 0) else {}
                qp_context = model_module.build_qp_context(policy_obs, hparams, **context_kwargs)
                env_action, logp, _entropy, value, policy_action, qp_diagnostics = (
                    model_module.get_action_and_value(
                        policy_params,
                        hparams,
                        policy_obs,
                        noise=noise,
                        return_policy_action=True,
                        return_diagnostics=True,
                        qp_context=qp_context,
                    )
                )
            else:
                env_action, logp, _entropy, value, policy_action = model_module.get_action_and_value(
                    policy_params,
                    hparams,
                    policy_obs,
                    noise=noise,
                    return_policy_action=True,
                )

            executed_action = env_action
            if safety_filter is not None:
                obstacles = Obstacles(state.human_positions, state.human_vels,
                                      state.human_radii, state.human_mask)
                executed_action, filter_diagnostics = safety_filter(state.robot_state, env_action, obstacles)
            next_state, reward, done, metrics = step_fn(env_cfg, state, executed_action, env_key)
            if has_policy_qp:
                reward = reward - float(hparams.qp_residual_reward_weight) * qp_diagnostics["qp_slack"]
            if cbf_reward_weight > 0:
                shaping = _cbf_reward(env_action, executed_action, filter_diagnostics, cbf_reward_sigma)
                shaping["cbf_rl/weighted_reward"] = cbf_reward_weight * shaping["cbf_rl/reward"]
                reward = reward + shaping["cbf_rl/weighted_reward"]
            done_bool = done > 0.5
            next_ppo_returns, completed_ppo_return = _update_ppo_episode_returns(
                running_ppo_returns,
                reward,
                done_bool,
            )
            if need_timeout_values:
                final_frame = observe_fn(env_cfg, next_state)
                final_history = _update_gnn_obs_history(history, final_frame, jnp.zeros_like(done_bool))
                final_value = model_module.critic_value(value_params, hparams, final_history.reshape(batch_size, -1))
            next_state = _maybe_reset_done_envs(env_cfg, next_state, reset_key, done_bool, batch_size, reset_fn)
            next_frame = observe_fn(env_cfg, next_state)
            next_history = _update_gnn_obs_history(history, next_frame, done_bool)

            if has_policy_qp:
                transition = {
                    "obs": policy_obs,
                    "qp_obs": qp_context["qp_obs"],
                    "qp_topk_ids": qp_context["qp_topk_ids"],
                    "act": env_action,
                    "policy_act": policy_action,
                    "logp": logp,
                    "rew": reward,
                    "done": done,
                    "val": value,
                    "completed_return": jnp.where(done_bool, metrics["episode_return"], jnp.nan),
                    "completed_ppo_return": completed_ppo_return,
                    "completed_length": jnp.where(done_bool, metrics["episode_length"], jnp.nan),
                    "success": metrics["success"],
                    "collision": metrics["collision"],
                    "timeout": metrics["timeout"],
                    "min_clearance": metrics["min_clearance"],
                    "boundary_violation": metrics["boundary_violation"],
                    "robot_omega_delta": metrics["robot_omega_delta"],
                    "robot_omega_delta_penalty": metrics["robot_omega_delta_penalty"],
                    "qp_slack": qp_diagnostics["qp_slack"],
                    "qp_slack_max": qp_diagnostics["qp_slack_max"],
                    "qp_correction": qp_diagnostics["qp_correction"],
                    "r_safe": qp_diagnostics["r_safe"],
                }
                if "robot_state" in qp_context:
                    transition["qp_robot_state"] = qp_context["robot_state"]
                if "beta" in qp_diagnostics:
                    transition["beta"] = qp_diagnostics["beta"]
                if "alpha" in qp_diagnostics:
                    transition["alpha"] = qp_diagnostics["alpha"]
                if hparams.qp_full_diagnostics:
                    transition.update({
                        "nominal_cbf_residual": qp_diagnostics["nominal_cbf_residual"],
                        "nominal_cbf_residual_max": qp_diagnostics["nominal_cbf_residual_max"],
                        "safe_cbf_residual": qp_diagnostics["safe_cbf_residual"],
                        "safe_cbf_residual_max": qp_diagnostics["safe_cbf_residual_max"],
                    })
            else:
                transition = {
                    "obs": policy_obs,
                    "act": env_action,
                    "policy_act": policy_action,
                    "logp": logp,
                    "rew": reward,
                    "done": done,
                    "val": value,
                    "completed_return": jnp.where(done_bool, metrics["episode_return"], jnp.nan),
                    "completed_ppo_return": completed_ppo_return,
                    "completed_length": jnp.where(done_bool, metrics["episode_length"], jnp.nan),
                    "success": metrics["success"],
                    "collision": metrics["collision"],
                    "timeout": metrics["timeout"],
                    "min_clearance": metrics["min_clearance"],
                    "boundary_violation": metrics["boundary_violation"],
                    "robot_omega_delta": metrics["robot_omega_delta"],
                    "robot_omega_delta_penalty": metrics["robot_omega_delta_penalty"],
                }
            if safety_filter is not None:
                transition['executed_act'] = executed_action
                transition.update({name: value for name, value in filter_diagnostics.items()
                                   if name.startswith('filter/')})
            if cbf_reward_weight > 0:
                transition.update(shaping)
            if need_timeout_values:
                truncated = metrics.get("truncated", metrics["timeout"])
                terminated = metrics.get("terminated", jnp.where(truncated > 0.5, 0.0, done))
                transition.update(terminated=terminated, truncated=truncated, next_values=final_value)
            if "physical_state_limit" in metrics:
                transition["control/physical_state_limit"] = metrics["physical_state_limit"].astype(jnp.float32)
            if "physical_contact" in metrics:
                transition["control/physical_contact"] = metrics["physical_contact"]
            return (next_state, next_history, next_ppo_returns, key), transition

        def value_fn(history):
            return model_module.critic_value(value_params, hparams, history.reshape((batch_size, -1)))

        initial = (nav_state, obs_history, ppo_episode_returns, rng)
        reward_metrics = {}
        if reward_loss_fn is None:
            final, transitions = jax.lax.scan(
                lambda carry, _: body(params, carry, None), initial, None, length=int(steps_per_env))
        else:
            final, transitions, reward_metrics = _collect_reward_windows(
                body, initial, params, value_fn, reward_loss_fn, int(reward_horizon), int(steps_per_env))
        next_state, next_history, next_ppo_returns, next_rng = final
        next_value = value_fn(next_history)
        beta_budget = (
            float(hparams.beta_max)
            if has_policy_qp and hasattr(hparams, "beta_max")
            else None
        )
        summary = _rollout_summary(transitions, has_policy_qp, beta_budget)
        if has_policy_qp:
            batch = {
                "obs": flat_time_batch(transitions["obs"]),
                "qp_obs": flat_time_batch(transitions["qp_obs"]),
                "qp_topk_ids": flat_time_batch(transitions["qp_topk_ids"]),
                "act": flat_time_batch(transitions["act"]),
                "policy_act": flat_time_batch(transitions["policy_act"]),
                "logp": flat_time_batch(transitions["logp"]),
                "rew": transitions["rew"],
                "done": transitions["done"],
                "val": transitions["val"],
                "next_value": next_value,
                "global_step": jnp.asarray(global_step, dtype=jnp.int32) + int(steps_per_env) * batch_size,
                **summary,
            }
        else:
            batch = {
                "obs": flat_time_batch(transitions["obs"]),
                "act": flat_time_batch(transitions["act"]),
                "policy_act": flat_time_batch(transitions["policy_act"]),
                "logp": flat_time_batch(transitions["logp"]),
                "rew": transitions["rew"],
                "done": transitions["done"],
                "val": transitions["val"],
                "next_value": next_value,
                "global_step": jnp.asarray(global_step, dtype=jnp.int32) + int(steps_per_env) * batch_size,
                **summary,
            }
        if safety_filter is not None:
            batch['executed_act'] = flat_time_batch(transitions['executed_act'])
            batch.update({name: (jnp.max(value) if name.endswith('_max') else jnp.mean(value))
                          for name, value in transitions.items() if name.startswith('filter/')})
        if "qp_robot_state" in transitions:
            batch["qp_robot_state"] = flat_time_batch(transitions["qp_robot_state"])
        if timeout_aware:
            batch.update({key: transitions[key] for key in ("terminated", "truncated", "next_values")})
        batch.update({key: jnp.mean(value) for key, value in transitions.items() if key.startswith("control/")})
        if cbf_reward_weight > 0:
            batch.update({name: jnp.mean(value) for name, value in transitions.items()
                          if name.startswith("cbf_rl/")})
        batch.update(reward_metrics)
        return batch, (next_state, next_history, next_ppo_returns), next_rng

    return collect
