"""Differentiable DiffCVaR slack-QP and diagnostics for supported robot dynamics."""

from __future__ import annotations

from model.qp_common import (_single_integrator_qp_objective, _unicycle_qp_objective, _solve_slack_qp, _masked_positive_stats, _masked_raw_stats)

import math

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy.special as jsp



def _normal_ppf(p):
    return math.sqrt(2.0) * jsp.erfinv(2.0 * p - 1.0)


def _normal_pdf(z):
    return (1.0 / math.sqrt(2.0 * math.pi)) * jnp.exp(-0.5 * z * z)


def _cvar_coeff_from_beta(beta, eps=1e-6):
    beta = jnp.clip(beta, eps, 1.0 - eps)
    return _normal_pdf(_normal_ppf(1.0 - beta)) / beta


def _extract_blocks(obs):
    batch = obs.shape[0]
    if obs.shape[1] <= 6:
        rel = jnp.zeros((batch, 1, 2), dtype=obs.dtype)
        vel = jnp.zeros_like(rel)
        radius = jnp.zeros((batch, 1), dtype=obs.dtype)
        mask = jnp.zeros((batch, 1), dtype=obs.dtype)
        return rel, vel, radius, mask
    blocks = obs[:, 6:].reshape((batch, -1, 6))
    return blocks[:, :, 0:2], blocks[:, :, 2:4], blocks[:, :, 4], jnp.clip(blocks[:, :, 5], 0.0, 1.0)


def _predict_gmm_multi(params, hparams, human_vel):
    batch, count, _dim = human_vel.shape
    flat_vel = human_vel.reshape((-1, 2))
    speed = jnp.linalg.norm(flat_vel, axis=1, keepdims=True)
    eps = jnp.asarray(1e-6, dtype=flat_vel.dtype)
    direction = flat_vel / (speed + eps)
    left = jnp.stack([-direction[:, 1], direction[:, 0]], axis=1)
    lateral = hparams.gmm_lateral_ratio * speed

    mu0 = flat_vel
    mu_left = flat_vel + lateral * left
    mu_right = flat_vel - lateral * left

    mu_left_norm = jnp.linalg.norm(mu_left, axis=1, keepdims=True)
    mu_right_norm = jnp.linalg.norm(mu_right, axis=1, keepdims=True)
    scale_left = jnp.where(mu_left_norm > eps, speed / mu_left_norm, jnp.ones_like(mu_left_norm))
    scale_right = jnp.where(mu_right_norm > eps, speed / mu_right_norm, jnp.ones_like(mu_right_norm))
    mu_left = mu_left * scale_left
    mu_right = mu_right * scale_right

    means = jnp.stack([mu0, mu_left, mu_right], axis=1)
    collapsed = jnp.broadcast_to(flat_vel[:, None, :], means.shape)
    means = jnp.where((speed[:, 0] < eps)[:, None, None], collapsed, means)

    gmm_weights = jax.lax.stop_gradient(params["gmm_weights"])
    gmm_variances = jax.lax.stop_gradient(params["gmm_variances"])
    if int(gmm_weights.shape[0]) != 3:
        raise ValueError("DiffCVaR JAX predictor expects three GMM components")
    variances = jnp.broadcast_to(gmm_variances[None, :], (flat_vel.shape[0], 3))
    return means.reshape((batch, count, 3, 2)), variances.reshape((batch, count, 3))


def _unicycle_cbf_constraints(actor_params, hparams, obs, beta, r_safe):
    batch = obs.shape[0]
    theta = obs[:, 4]
    rel, human_vel, _obstacle_radius, mask = _extract_blocks(obs)

    eps = hparams.lookahead_distance
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    heading = jnp.stack([cos_t, sin_t], axis=1)[:, None, :]
    lookahead_rel = rel + eps * heading

    r_safe = jnp.asarray(r_safe, dtype=obs.dtype)
    if r_safe.ndim == 1:
        r_safe = r_safe[:, None]
    r_safe = r_safe + eps
    h = jnp.sum(lookahead_rel * lookahead_rel, axis=2) - r_safe * r_safe

    lg_v = 2.0 * (lookahead_rel[:, :, 0] * cos_t[:, None] + lookahead_rel[:, :, 1] * sin_t[:, None])
    lg_w = 2.0 * eps * (lookahead_rel[:, :, 1] * cos_t[:, None] - lookahead_rel[:, :, 0] * sin_t[:, None])

    means, variances = _predict_gmm_multi(actor_params, hparams, human_vel)
    cvar_coeff = _cvar_coeff_from_beta(jnp.asarray(beta, dtype=obs.dtype))
    if cvar_coeff.ndim == 1:
        cvar_coeff = cvar_coeff.reshape((batch, 1, 1))
    else:
        cvar_coeff = cvar_coeff[:, :, None]
    sigma_f = jnp.sqrt(4.0 * variances * jnp.sum(lookahead_rel * lookahead_rel, axis=2, keepdims=True) + 1e-8)
    rel_dot_mu = 2.0 * jnp.sum(means * lookahead_rel[:, :, None, :], axis=3)
    rhs = -hparams.alpha * h[:, :, None] + rel_dot_mu + sigma_f * cvar_coeff
    rhs_worst = 0.1 * jnn.logsumexp(rhs / 0.1, axis=2)

    g = jnp.stack([-(lg_v * mask), -(lg_w * mask)], axis=2)
    h_qp = -(rhs_worst * mask)
    return g, h_qp, mask


def _single_integrator_cbf_constraints(actor_params, hparams, obs, beta, r_safe):
    batch = obs.shape[0]
    rel, human_vel, _obstacle_radius, mask = _extract_blocks(obs)

    r_safe = jnp.asarray(r_safe, dtype=obs.dtype)
    if r_safe.ndim == 1:
        r_safe = r_safe[:, None]
    h = jnp.sum(rel * rel, axis=2) - r_safe * r_safe

    means, variances = _predict_gmm_multi(actor_params, hparams, human_vel)
    cvar_coeff = _cvar_coeff_from_beta(jnp.asarray(beta, dtype=obs.dtype))
    if cvar_coeff.ndim == 1:
        cvar_coeff = cvar_coeff.reshape((batch, 1, 1))
    else:
        cvar_coeff = cvar_coeff[:, :, None]
    sigma_f = jnp.sqrt(4.0 * variances * jnp.sum(rel * rel, axis=2, keepdims=True) + 1e-8)
    rel_dot_mu = 2.0 * jnp.sum(means * rel[:, :, None, :], axis=3)
    rhs = -hparams.alpha * h[:, :, None] + rel_dot_mu + sigma_f * cvar_coeff
    rhs_worst = 0.1 * jnn.logsumexp(rhs / 0.1, axis=2)

    g = -2.0 * rel * mask[:, :, None]
    h_qp = -(rhs_worst * mask)
    return g, h_qp, mask


def _cbf_constraints(actor_params, hparams, obs, beta, r_safe):
    if hparams.robot_type == "single_integrator":
        return _single_integrator_cbf_constraints(actor_params, hparams, obs, beta, r_safe)
    return _unicycle_cbf_constraints(actor_params, hparams, obs, beta, r_safe)


def solve_unicycle_slack_qp(params, hparams, obs, nominal_xy, beta, r_safe):
    q, p = _unicycle_qp_objective(hparams, obs, nominal_xy)
    g, h_qp, _mask = _unicycle_cbf_constraints(params["actor"], hparams, obs, beta, r_safe)
    return _solve_slack_qp(params, hparams, obs, q, p, g, h_qp)


def solve_single_integrator_slack_qp(params, hparams, obs, nominal_xy, beta, r_safe):
    q, p = _single_integrator_qp_objective(hparams, obs, nominal_xy)
    g, h_qp, _mask = _single_integrator_cbf_constraints(
        params["actor"], hparams, obs, beta, r_safe
    )
    return _solve_slack_qp(params, hparams, obs, q, p, g, h_qp)


def solve_slack_qp(params, hparams, obs, nominal_xy, beta, r_safe):
    if hparams.robot_type == "single_integrator":
        return solve_single_integrator_slack_qp(
            params, hparams, obs, nominal_xy, beta, r_safe
        )
    return solve_unicycle_slack_qp(params, hparams, obs, nominal_xy, beta, r_safe)


def lightweight_diagnostics(params, hparams, obs, nominal_xy, beta, r_safe, safe_xy, slack):
    del params, hparams
    _rel, _human_vel, _obstacle_radius, mask = _extract_blocks(obs)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "beta": beta.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
    }


def full_diagnostics(params, hparams, obs, nominal_xy, beta, r_safe, safe_xy, slack):
    g, h_qp, mask = _cbf_constraints(params["actor"], hparams, obs, beta, r_safe)
    nominal_raw = jnp.sum(g * nominal_xy[:, None, :], axis=2) - h_qp
    safe_raw = jnp.sum(g * safe_xy[:, None, :], axis=2) - h_qp
    nominal_mean, nominal_max = _masked_positive_stats(nominal_raw, mask)
    safe_mean, safe_max = _masked_positive_stats(safe_raw, mask)
    diagnostics = lightweight_diagnostics(params, hparams, obs, nominal_xy, beta, r_safe, safe_xy, slack)
    diagnostics.update({
        "nominal_cbf_residual": nominal_mean.astype(jnp.float32),
        "nominal_cbf_residual_max": nominal_max.astype(jnp.float32),
        "safe_cbf_residual": safe_mean.astype(jnp.float32),
        "safe_cbf_residual_max": safe_max.astype(jnp.float32),
    })
    return diagnostics
