"""Quadratically penalized obstacle relaxation; physical inputs remain bounded."""
import math
import numpy as np
import jax.numpy as jnp

from env.robots import UnicycleRobot
from .cbf_qp import (
    SolverConfig, _tracking_qp_data, _solve_qp_data,
    _nominal_control, _cbf_constraints, _control_output_matrix,
)


def validate_slack_weight(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("controller.slack_weight must be finite and positive")


def solve_slack_qp(nominal, constraints, low, high, weights, *, soft_mask,
                   slack_weight, solver=SolverConfig(), output_matrix=None):
    """Minimize tracking cost + rho/2*||delta||^2 with A*u+b+E*delta>=0.

    Only rows selected by soft_mask receive nonnegative, unbounded slacks.
    Returned slack entries retain the original constraint-row ordering.
    valid checks the relaxed QP; residual still reports the original conditions.
    """
    validate_slack_weight(slack_weight)
    nominal, low, high, cbf_A, cbf_b, tracking_P, control_scale = _tracking_qp_data(
        nominal, constraints, low, high, weights, output_matrix)
    n, d = cbf_A.shape
    mask = np.asarray(soft_mask)
    if mask.shape != (n,) or not np.isfinite(mask).all():
        raise ValueError("soft_mask must be a finite vector with one entry per CBF row")
    ids = np.flatnonzero(mask > .5)
    k = len(ids)
    E = np.zeros((n, k))
    E[ids, np.arange(k)] = 1.
    A = np.block([
        [cbf_A, E],
        [np.eye(d), np.zeros((d, k))],
        [-np.eye(d), np.zeros((d, k))],
        [np.zeros((k, d)), np.eye(k)],
    ])
    b = np.concatenate([cbf_b, -low, high, np.zeros(k)])
    # delta = z_delta/sqrt(rho): condition the cost without changing its units.
    scale = np.concatenate([control_scale, np.full(k, 1./math.sqrt(slack_weight))])
    P = np.zeros((d+k, d+k))
    P[:d, :d] = tracking_P
    P[d:, d:] = np.eye(k)
    augmented = _solve_qp_data(
        np.concatenate([nominal, np.zeros(k)]), A, b, P, scale,
        constraints.domain, solver, solver.feasibility_tolerance)
    candidate = augmented.candidate[:d]
    raw = jnp.concatenate([
        jnp.asarray(cbf_A)@candidate+jnp.asarray(cbf_b),
        candidate-jnp.asarray(low), jnp.asarray(high)-candidate])
    result = augmented._replace(
        action=augmented.action[:d], candidate=candidate, residual=raw,
        correction_norm=float(np.linalg.norm(np.asarray(candidate)-nominal)))
    slack = jnp.zeros(n, dtype=candidate.dtype).at[ids].set(augmented.candidate[d:])
    return result, slack


def generate_action(robot, state, goal, obstacles, control_config, *, slack_weight):
    """Return [v, omega] and separate QP/slack diagnostics for a unicycle."""
    if not isinstance(robot, UnicycleRobot):
        raise ValueError("cbf_qp_slack currently supports unicycle only")
    nominal = _nominal_control(robot, state, goal, control_config)
    constraints = _cbf_constraints(robot, state, obstacles, control_config)
    low, high = robot.input_bounds()
    result, slack = solve_slack_qp(
        nominal, constraints, low, high, control_config.weights,
        soft_mask=obstacles.mask, slack_weight=slack_weight,
        solver=control_config.solver,
        output_matrix=_control_output_matrix(robot, state, control_config))
    return result.action, {"nominal": nominal, "qp": result, "slack": slack}
