"""Differentiable deterministic CBF slack-QP for supported robot dynamics."""

from __future__ import annotations

from model.qp_common import (_single_integrator_qp_objective, _unicycle_qp_objective, _solve_slack_qp, _masked_positive_stats, _masked_raw_stats)

import jax.numpy as jnp



def _extract_blocks(obs):
    batch = obs.shape[0]
    blocks = obs[:, 6:].reshape((batch, -1, 6))
    return (
        blocks[:, :, 0:2],
        blocks[:, :, 2:4],
        jnp.clip(blocks[:, :, 5], 0.0, 1.0),
    )


def _unicycle_cbf_constraints(hparams, obs, alpha, r_safe):
    theta = obs[:, 4]
    rel, human_vel, mask = _extract_blocks(obs)

    eps = hparams.lookahead_distance
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    heading = jnp.stack([cos_t, sin_t], axis=1)[:, None, :]
    lookahead_rel = rel + eps * heading

    alpha = jnp.asarray(alpha, dtype=obs.dtype)
    r_safe = jnp.asarray(r_safe, dtype=obs.dtype) + eps
    h = jnp.sum(lookahead_rel * lookahead_rel, axis=2) - r_safe * r_safe

    lg_v = 2.0 * (
        lookahead_rel[:, :, 0] * cos_t[:, None]
        + lookahead_rel[:, :, 1] * sin_t[:, None]
    )
    lg_w = 2.0 * eps * (
        lookahead_rel[:, :, 1] * cos_t[:, None]
        - lookahead_rel[:, :, 0] * sin_t[:, None]
    )
    rel_dot_human_vel = 2.0 * jnp.sum(human_vel * lookahead_rel, axis=2)
    rhs = -alpha * h + rel_dot_human_vel

    g = jnp.stack([-(lg_v * mask), -(lg_w * mask)], axis=2)
    h_qp = -(rhs * mask)
    return g, h_qp, mask


def _single_integrator_cbf_constraints(hparams, obs, alpha, r_safe):
    del hparams
    rel, human_vel, mask = _extract_blocks(obs)
    alpha = jnp.asarray(alpha, dtype=obs.dtype)
    r_safe = jnp.asarray(r_safe, dtype=obs.dtype)
    h = jnp.sum(rel * rel, axis=2) - r_safe * r_safe
    rel_dot_human_vel = 2.0 * jnp.sum(human_vel * rel, axis=2)
    rhs = -alpha * h + rel_dot_human_vel

    g = -2.0 * rel * mask[:, :, None]
    h_qp = -(rhs * mask)
    return g, h_qp, mask


def _cbf_constraints(hparams, obs, alpha, r_safe):
    if hparams.robot_type == "single_integrator":
        return _single_integrator_cbf_constraints(hparams, obs, alpha, r_safe)
    return _unicycle_cbf_constraints(hparams, obs, alpha, r_safe)


def solve_slack_qp(params, hparams, obs, nominal_xy, alpha, r_safe):
    if hparams.robot_type == "single_integrator":
        q, p = _single_integrator_qp_objective(hparams, obs, nominal_xy)
    else:
        q, p = _unicycle_qp_objective(hparams, obs, nominal_xy)
    g, h_qp, _mask = _cbf_constraints(hparams, obs, alpha, r_safe)
    return _solve_slack_qp(params, hparams, obs, q, p, g, h_qp)


def lightweight_diagnostics(
    hparams, obs, nominal_xy, alpha, r_safe, safe_xy, slack
):
    del hparams
    _rel, _human_vel, mask = _extract_blocks(obs)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    zero = jnp.zeros((obs.shape[0],), dtype=obs.dtype)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_penalty": slack_mean.astype(jnp.float32),
        "qp_penalty_max": slack_max.astype(jnp.float32),
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "nominal_cbf_residual": zero.astype(jnp.float32),
        "nominal_cbf_residual_max": zero.astype(jnp.float32),
        "safe_cbf_residual": zero.astype(jnp.float32),
        "safe_cbf_residual_max": zero.astype(jnp.float32),
        "alpha": alpha.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
        "nominal_action_norm": jnp.linalg.norm(nominal_xy, axis=1).astype(
            jnp.float32
        ),
        "safe_action_norm": jnp.linalg.norm(safe_xy, axis=1).astype(jnp.float32),
    }


def full_diagnostics(hparams, obs, nominal_xy, alpha, r_safe, safe_xy, slack):
    g, h_qp, mask = _cbf_constraints(hparams, obs, alpha, r_safe)
    nominal_raw = jnp.sum(g * nominal_xy[:, None, :], axis=2) - h_qp
    safe_raw = jnp.sum(g * safe_xy[:, None, :], axis=2) - h_qp
    nominal_mean, nominal_max = _masked_positive_stats(nominal_raw, mask)
    safe_mean, safe_max = _masked_positive_stats(safe_raw, mask)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_penalty": slack_mean.astype(jnp.float32),
        "qp_penalty_max": slack_max.astype(jnp.float32),
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "nominal_cbf_residual": nominal_mean.astype(jnp.float32),
        "nominal_cbf_residual_max": nominal_max.astype(jnp.float32),
        "safe_cbf_residual": safe_mean.astype(jnp.float32),
        "safe_cbf_residual_max": safe_max.astype(jnp.float32),
        "alpha": alpha.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
        "nominal_action_norm": jnp.linalg.norm(nominal_xy, axis=1).astype(
            jnp.float32
        ),
        "safe_action_norm": jnp.linalg.norm(safe_xy, axis=1).astype(jnp.float32),
    }
