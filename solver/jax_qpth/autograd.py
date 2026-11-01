"""qpth-compatible implicit differentiation for the JAX QP layer."""

from __future__ import annotations

import jax.numpy as jnp

from .pdipm import factor_kkt, pre_factor_kkt, solve_kkt_factored
from .util import bger


def qpth_backward(Q, p, G, h, A, b, zhats, lams, slacks, nus, dl_dzhat):
    """Return qpth-style gradients for Q, p, G, h, A, b."""
    del h
    # Failed forward rows are masked by the caller. Do not factor their invalid
    # KKT systems even when their cotangent is zero (0 * NaN is still NaN).
    active = jnp.any(dl_dzhat != 0., axis=-1)
    matrix_mask, vector_mask = active[:, None, None], active[:, None]
    Q = jnp.where(matrix_mask, Q, jnp.eye(Q.shape[-1], dtype=Q.dtype))
    G = jnp.where(matrix_mask, G, 0.)
    A = jnp.where(matrix_mask, A, jnp.eye(A.shape[-2], A.shape[-1], dtype=A.dtype))
    zhats = jnp.where(vector_mask, zhats, 0.)
    lams = jnp.where(vector_mask, lams, 1.)
    slacks = jnp.where(vector_mask, slacks, 1.)
    nus = jnp.where(vector_mask, nus, 0.)
    d = jnp.clip(lams, min=1e-8) / jnp.clip(slacks, min=1e-8)
    n_batch, nineq, _nz = G.shape
    neq = A.shape[1]
    zeros_ineq = jnp.zeros((n_batch, nineq), dtype=G.dtype)
    zeros_eq = jnp.zeros((n_batch, neq), dtype=G.dtype)
    kkt_factors = pre_factor_kkt(Q, G, A)
    S_LU = factor_kkt(kkt_factors, d, neq)
    dx, _ds, dlam, dnu = solve_kkt_factored(kkt_factors, d, G, A, S_LU, dl_dzhat, zeros_ineq, zeros_ineq, zeros_eq)

    dps = dx
    dGs = bger(dlam, zhats) + bger(lams, dx)
    dhs = -dlam
    dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
    if neq > 0:
        dAs = bger(dnu, zhats) + bger(nus, dx)
        dbs = -dnu
    else:
        dAs = jnp.zeros_like(A)
        dbs = jnp.zeros_like(b)
    return tuple(jnp.where(active.reshape((active.shape[0],)+(1,)*(v.ndim-1)), v, 0.)
                 for v in (dQs, dps, dGs, dhs, dAs, dbs))
