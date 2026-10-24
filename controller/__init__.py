"""CBF-QP controller factory."""
from dataclasses import fields
from functools import partial
import math

from omegaconf import OmegaConf

from env.robots import make_robot
from env.robots.base import Obstacles
from .cbf_qp import ControlResult, SolverConfig, generate_action, validate_slack_weight

__all__ = ["Obstacles", "ControlResult", "SolverConfig", "generate_action", "make_controller"]


def _required_section(node, names, label):
    values = OmegaConf.to_container(node, resolve=True)
    missing = set(names)-values.keys()
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
    return values


def make_controller(config):
    """Return (robot, solver, action_fn)."""
    if config.controller.type != "cbf_qp":
        raise ValueError(f"unsupported controller type: {config.controller.type!r}")
    top_k = config.controller.qp_obs_top_k
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError("controller.qp_obs_top_k must be null or a positive integer")
    weight = config.controller.slack_weight
    validate_slack_weight(weight)
    if weight > 0 and config.robot.type != "unicycle":
        raise ValueError("positive slack_weight currently supports unicycle only")
    dt = float(config.env.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("env.dt must be positive")
    if config.robot.type not in {"unicycle", "double_integrator", "quadrotor", "humanoid"}:
        raise ValueError("controller robot must be unicycle, double_integrator, quadrotor or humanoid")
    robot = make_robot(config)
    values = _required_section(config.controller.solver,
                               [field.name for field in fields(SolverConfig)], "controller.solver")
    solver = SolverConfig(**values)
    action_fn = partial(generate_action, robot, dt=dt, solver=solver, slack_weight=weight,
                        qp_obs_top_k=top_k)
    return robot, solver, action_fn
