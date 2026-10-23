"""Shared QP objective, slack assembly, bounds and diagnostic reductions."""

from __future__ import annotations

import jax.numpy as jnp
from solver.jax_qpth import QPFunction


def _single_integrator_qp_objective(hparams, obs, nominal_xy):
    batch = obs.shape[0]
    eye = jnp.eye(hparams.act_dim, dtype=obs.dtype)[None, :, :]
    q = 2.0 * jnp.broadcast_to(eye, (batch, hparams.act_dim, hparams.act_dim))
    p = -2.0 * nominal_xy
    return q, p


def _unicycle_qp_objective(hparams, obs, nominal_xy):
    batch = obs.shape[0]
    dtype = obs.dtype
    theta = obs[:, 4]
    eps = hparams.lookahead_distance
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    j_mat = jnp.stack(
        [
            jnp.stack([cos_t, -eps * sin_t], axis=1),
            jnp.stack([sin_t, eps * cos_t], axis=1),
        ],
        axis=1,
    )
    jt = jnp.swapaxes(j_mat, 1, 2)
    eye = jnp.eye(hparams.act_dim, dtype=dtype)[None, :, :]
    q = 2.0 * jnp.matmul(jt, j_mat) + 1e-6 * eye
    p = -2.0 * jnp.matmul(jt, nominal_xy[:, :, None]).squeeze(-1)
    return q, p


def _solve_slack_qp(params, hparams, obs, q, p, g, h_qp):
    batch = obs.shape[0]
    dtype = obs.dtype
    slack_dim = g.shape[1]
    slack_eye = jnp.eye(slack_dim, dtype=dtype)[None, :, :]
    slack_eye = jnp.broadcast_to(slack_eye, (batch, slack_dim, slack_dim))
    zeros_u_slack = jnp.zeros(
        (batch, hparams.act_dim, slack_dim), dtype=dtype
    )
    zeros_slack_u = jnp.zeros(
        (batch, slack_dim, hparams.act_dim), dtype=dtype
    )

    slack_q = 2.0 * hparams.qp_slack_weight * slack_eye
    q_top = jnp.concatenate([q, zeros_u_slack], axis=2)
    q_bottom = jnp.concatenate([zeros_slack_u, slack_q], axis=2)
    q_qp = jnp.concatenate([q_top, q_bottom], axis=1)
    p_qp = jnp.concatenate(
        [p, jnp.zeros((batch, slack_dim), dtype=dtype)], axis=1
    )

    cbf_g = jnp.concatenate([g, -slack_eye], axis=2)
    slack_g = jnp.concatenate([zeros_slack_u, -slack_eye], axis=2)
    g_qp = jnp.concatenate([cbf_g, slack_g], axis=1)
    h_qp_full = jnp.concatenate(
        [h_qp, jnp.zeros((batch, slack_dim), dtype=dtype)], axis=1
    )

    total_dim = q_qp.shape[1]
    bound_eye = jnp.eye(hparams.act_dim, dtype=dtype)
    bound_pad = jnp.zeros(
        (hparams.act_dim, total_dim - hparams.act_dim), dtype=dtype
    )
    upper_g = jnp.concatenate([bound_eye, bound_pad], axis=1)
    lower_g = jnp.concatenate([-bound_eye, bound_pad], axis=1)
    bound_g = jnp.concatenate([upper_g, lower_g], axis=0)
    bound_g = jnp.broadcast_to(
        bound_g[None, :, :], (batch, 2 * hparams.act_dim, total_dim)
    )

    action_low = jnp.asarray(params["action_low"], dtype=dtype)
    action_high = jnp.asarray(params["action_high"], dtype=dtype)
    bound_h = jnp.concatenate([action_high, -action_low], axis=0)
    bound_h = jnp.broadcast_to(
        bound_h[None, :], (batch, 2 * hparams.act_dim)
    )
    g_qp = jnp.concatenate([g_qp, bound_g], axis=1)
    h_qp_full = jnp.concatenate([h_qp_full, bound_h], axis=1)

    qp_dtype = jnp.float64
    empty = jnp.empty((0,), dtype=qp_dtype)
    solution = QPFunction(
        eps=hparams.qp_eps,
        verbose=hparams.qp_verbose,
        notImprovedLim=hparams.qp_not_improved_lim,
        maxIter=hparams.qp_max_iter,
        check_Q_spd=hparams.qp_check_q_spd,
    )(
        q_qp.astype(qp_dtype),
        p_qp.astype(qp_dtype),
        g_qp.astype(qp_dtype),
        h_qp_full.astype(qp_dtype),
        empty,
        empty,
    )
    action = solution[:, : hparams.act_dim].astype(dtype)
    slack = solution[
        :, hparams.act_dim : hparams.act_dim + slack_dim
    ].astype(dtype)
    return action, jnp.maximum(slack, 0.0)


def _masked_positive_stats(values, mask):
    values = jnp.where(mask > 0.5, jnp.maximum(values, 0.0), 0.0)
    denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    return jnp.sum(values, axis=1) / denom, jnp.max(values, axis=1)


def _masked_raw_stats(values, mask):
    values = jnp.where(mask > 0.5, values, 0.0)
    denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    return jnp.sum(values, axis=1) / denom, jnp.max(values, axis=1)
