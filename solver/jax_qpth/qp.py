"""qpth-like dense QPFunction implemented with JAX and custom VJP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError
import numpy as np

from .autograd import qpth_backward
from .pdipm import forward as pdipm_forward
from .util import expand_qp_params, reduce_grad


def qp_residuals(Q, p, G, h, x, dual, *, A=None, b=None, nu=None,
                 slack=None, xp=jnp):
    """Absolute KKT errors in the original QP coordinates, including all slacks.

    Order: primal feasibility, stationarity, complementarity, dual feasibility.
    Undo solver variable/row scaling before calling. Nonfinite data or solutions
    give infinite errors; positive decision-variable slack is allowed by G/h.
    """
    maximum = lambda value: xp.max(value, axis=-1, initial=0.)
    margin = h - (G @ x[..., None])[..., 0]
    s = margin if slack is None else slack
    primal = xp.maximum(maximum(-margin), maximum(-s))
    primal = xp.maximum(primal, maximum(xp.abs(s-margin)))
    stationarity = (Q @ x[..., None])[..., 0] + p + (xp.swapaxes(G, -1, -2) @ dual[..., None])[..., 0]
    values = [(Q, 2), (p, 1), (G, 2), (h, 1), (x, 1), (dual, 1), (s, 1)]
    if A is not None:
        primal = xp.maximum(primal, maximum(xp.abs((A @ x[..., None])[..., 0]-b)))
        stationarity = stationarity + (xp.swapaxes(A, -1, -2) @ nu[..., None])[..., 0]
        values += [(A, 2), (b, 1), (nu, 1)]
    errors = xp.stack([primal, maximum(xp.abs(stationarity)),
                       xp.maximum(maximum(xp.abs(margin*dual)), maximum(xp.abs(s*dual))),
                       maximum(-dual)], axis=-1)
    finite = xp.all(xp.isfinite(errors), axis=-1)
    for value, ndim in values:
        finite = finite & xp.all(xp.isfinite(value), axis=tuple(range(-ndim, 0)))
    return xp.where(finite[..., None], errors, xp.inf)


def qp_solution_valid(errors, tolerance=1e-6):
    """Per-problem acceptance; failure does not certify infeasibility."""
    errors = jax.lax.stop_gradient(errors)
    return jnp.all(jnp.isfinite(errors) & (errors <= tolerance), axis=-1)


def check_qp_residuals(errors, *, tolerance=1e-6, label="QP"):
    """Reject invalid JAX solutions without altering the successful action VJP."""
    errors = jax.lax.stop_gradient(errors)
    valid = qp_solution_valid(errors, tolerance)
    def fail(_):
        def raise_failure(ok, residuals):
            if not np.all(ok):
                failed = np.flatnonzero(~np.asarray(ok)).tolist()
                raise RuntimeError(f"{label} failed for batch rows {failed}; "
                                   f"KKT residuals={np.asarray(residuals).tolist()}")
        jax.debug.callback(raise_failure, valid, errors, ordered=True)
        return 0
    jax.lax.cond(jnp.all(valid), lambda _: 0, fail, None)


def _check_q_spd(Q):
    chol = jnp.linalg.cholesky(Q)
    ok = jnp.all(jnp.isfinite(chol))
    try:
        ok_bool = bool(ok)
    except TracerBoolConversionError:
        def _raise_if_not_spd(value):
            if not bool(np.asarray(value)):
                raise RuntimeError("Q is not SPD.")

        jax.debug.callback(_raise_if_not_spd, ok)
        return
    if not ok_bool:
        raise RuntimeError("Q is not SPD.")


def solve_qp(Q, p, G, h, A, b, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, check_Q_spd=True, safe_step=False):
    expanded = expand_qp_params(Q, p, G, h, A, b)
    if expanded.A.shape[1] == 0 and expanded.G.shape[1] == 0:
        raise AssertionError("at least one equality or inequality constraint is required")
    if check_Q_spd:
        _check_q_spd(expanded.Q)
    return pdipm_forward(
        expanded.Q,
        expanded.p,
        expanded.G,
        expanded.h,
        expanded.A,
        expanded.b,
        eps=eps,
        verbose=verbose,
        notImprovedLim=notImprovedLim,
        maxIter=maxIter,
        safe_step=safe_step,
    )


def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, check_Q_spd=True, safe_step=False, return_duals=False):
    """Differentiable dense QP; optional dual/slack outputs are diagnostics only."""

    @jax.custom_vjp
    def _solve(Q, p, G, h, A, b):
        result = solve_qp(
            Q,
            p,
            G,
            h,
            A,
            b,
            eps=eps,
            verbose=verbose,
            notImprovedLim=notImprovedLim,
            maxIter=maxIter,
            safe_step=safe_step,
            check_Q_spd=check_Q_spd,
        )
        return result if return_duals else result[0]

    def _solve_fwd(Q, p, G, h, A, b):
        expanded = expand_qp_params(Q, p, G, h, A, b)
        if check_Q_spd:
            _check_q_spd(expanded.Q)
        zhats, nus, lams, slacks = pdipm_forward(
            expanded.Q,
            expanded.p,
            expanded.G,
            expanded.h,
            expanded.A,
            expanded.b,
            eps=eps,
            verbose=verbose,
            notImprovedLim=notImprovedLim,
            maxIter=maxIter,
            safe_step=safe_step,
        )
        expanded_values = (expanded.Q, expanded.p, expanded.G, expanded.h, expanded.A, expanded.b)
        expanded_flags = (
            expanded.Q_expanded,
            expanded.p_expanded,
            expanded.G_expanded,
            expanded.h_expanded,
            expanded.A_expanded,
            expanded.b_expanded,
        )
        residual = (
            Q,
            p,
            G,
            h,
            A,
            b,
            expanded_values,
            expanded_flags,
            zhats,
            lams,
            slacks,
            nus,
        )
        result = (zhats, nus, lams, slacks) if return_duals else zhats
        return result, residual

    def _solve_bwd(residual, dl_dzhat):
        if return_duals:
            dl_dzhat = dl_dzhat[0]
        (
            Q_orig,
            p_orig,
            G_orig,
            h_orig,
            A_orig,
            b_orig,
            expanded_values,
            expanded_flags,
            zhats,
            lams,
            slacks,
            nus,
        ) = residual
        Q_exp, p_exp, G_exp, h_exp, A_exp, b_exp = expanded_values
        Q_e, p_e, G_e, h_e, A_e, b_e = expanded_flags
        dQs, dps, dGs, dhs, dAs, dbs = qpth_backward(
            Q_exp,
            p_exp,
            G_exp,
            h_exp,
            A_exp,
            b_exp,
            zhats,
            lams,
            slacks,
            nus,
            dl_dzhat,
        )
        return (
            reduce_grad(dQs, Q_orig, Q_e),
            reduce_grad(dps, p_orig, p_e),
            reduce_grad(dGs, G_orig, G_e),
            reduce_grad(dhs, h_orig, h_e),
            reduce_grad(dAs, A_orig, A_e),
            reduce_grad(dbs, b_orig, b_e),
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve
