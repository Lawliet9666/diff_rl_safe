"""Differentiable CBF slack-QP."""

from __future__ import annotations

from model.qp_common import (_single_integrator_qp_objective, _unicycle_qp_objective, _solve_slack_qp, _masked_positive_stats, _masked_raw_stats)

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from env.robots.base import Obstacles



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


def solve_slack_qp(params, hparams, obs, nominal_xy, alpha, r_safe, *, robot_state=None, return_status=False):
    if getattr(hparams, "robot_state_dim", 0):
        if robot_state is None:
            raise ValueError("High-order CBF-QP requires full-precision robot_state")
        robot = hparams.robot
        constraints, _mask = robot_cbf_constraints(hparams, obs, robot_state, alpha, r_safe)
        matrices = jax.vmap(robot.control_output_matrix)(robot_state)
        weighted = jnp.asarray(robot.weights, dtype=matrices.dtype)[None, :, None]*matrices
        q = 2*jnp.swapaxes(matrices, 1, 2) @ weighted
        p = -(q @ nominal_xy[..., None]).squeeze(-1)
        n = obs[:, 6:].shape[1]//6
        chol = jnp.linalg.cholesky(q)
        identity = jnp.broadcast_to(jnp.eye(hparams.act_dim, dtype=q.dtype), q.shape)
        transform = jsp_linalg.solve_triangular(jnp.swapaxes(chol, 1, 2), identity, lower=False)
        # Relax obstacle rows only; state/input limits stay hard.
        result = _solve_slack_qp(params, hparams, robot_state, q, p,
                              -constraints.A[:, :n], constraints.b[:, :n],
                              hard_g=-constraints.A[:, n:], hard_h=constraints.b[:, n:],
                              control_transform=transform, return_status=return_status)
        action, slack = result[:2]
        result = (action.astype(obs.dtype), slack.astype(obs.dtype), *result[2:])
        return result
    if hparams.robot_type == "single_integrator":
        q, p = _single_integrator_qp_objective(hparams, obs, nominal_xy)
    else:
        q, p = _unicycle_qp_objective(hparams, obs, nominal_xy)
    g, h_qp, _mask = _cbf_constraints(hparams, obs, alpha, r_safe)
    return _solve_slack_qp(params, hparams, obs, q, p, g, h_qp, return_status=return_status)


def lightweight_diagnostics(
    hparams, obs, nominal_xy, alpha, r_safe, safe_xy, slack, *, robot_state=None
):
    del hparams, robot_state
    _rel, _human_vel, mask = _extract_blocks(obs)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "alpha": alpha.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
    }


def full_diagnostics(hparams, obs, nominal_xy, alpha, r_safe, safe_xy, slack, *, robot_state=None):
    if getattr(hparams, "robot_state_dim", 0):
        constraints, mask = robot_cbf_constraints(hparams, obs, robot_state, alpha, r_safe)
        n = mask.shape[1]
        g, h_qp = -constraints.A[:, :n], constraints.b[:, :n]
    else:
        g, h_qp, mask = _cbf_constraints(hparams, obs, alpha, r_safe)
    nominal_raw = jnp.sum(g * nominal_xy[:, None, :], axis=2) - h_qp
    safe_raw = jnp.sum(g * safe_xy[:, None, :], axis=2) - h_qp
    nominal_mean, nominal_max = _masked_positive_stats(nominal_raw, mask)
    safe_mean, safe_max = _masked_positive_stats(safe_raw, mask)
    diagnostics = lightweight_diagnostics(hparams, obs, nominal_xy, alpha, r_safe, safe_xy, slack, robot_state=robot_state)
    diagnostics.update({
        "nominal_cbf_residual": nominal_mean.astype(jnp.float32),
        "nominal_cbf_residual_max": nominal_max.astype(jnp.float32),
        "safe_cbf_residual": safe_mean.astype(jnp.float32),
        "safe_cbf_residual_max": safe_max.astype(jnp.float32),
    })
    return diagnostics


def robot_cbf_constraints(hparams, obs, robot_state, kappa, r_safe):
    blocks = obs[:, 6:].reshape((obs.shape[0], -1, 6))
    centers = robot_state[:, None, :2] - blocks[..., :2].astype(robot_state.dtype)
    obstacles = Obstacles(centers, blocks[..., 2:4].astype(robot_state.dtype),
                          blocks[..., 4].astype(robot_state.dtype), blocks[..., 5])
    constraints = jax.vmap(lambda state, obstacle, rates, radius:
        hparams.robot.cbf_constraints(state, obstacle, hparams.control_dt,
                                      kappa=rates, r_safe=radius))(
            robot_state, obstacles, kappa, r_safe)
    return constraints, jnp.clip(blocks[..., 5], 0., 1.)
