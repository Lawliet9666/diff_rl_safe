"""Build robot parameters and a ready-to-use common CBF-QP filter."""
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


def _build_robot(config):
    """Merge current grouped YAML settings into one immutable robot snapshot."""
    r, c, e = config.robot, config.controller, config.env
    if r.type not in {"unicycle", "quadrotor", "humanoid"}:
        raise ValueError("controller robot must be unicycle, quadrotor or humanoid")
    label = f"controller.{r.type}"
    groups = ("nominal", "cbf", "qp")
    sections = _required_section(c[r.type], groups, label)
    params = {key: value for key, value in sections.items() if key not in groups}
    blocks = [sections["nominal"], sections["qp"]]
    if r.type == "humanoid":
        cbf = _required_section(OmegaConf.create(sections["cbf"]),
                                ("distance", "parabolic"), f"{label}.cbf")
        blocks.extend([cbf["distance"], cbf["parabolic"]])
    else:
        blocks.append(sections["cbf"])
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError(f"{label} parameter groups must be mappings")
        duplicate = params.keys() & block.keys()
        if duplicate:
            raise ValueError(f"{label} has duplicate parameters: {', '.join(sorted(duplicate))}")
        params.update(block)
    common = dict(margin=float(e.safety_margin))
    if r.type in {"unicycle", "humanoid"}:
        common["goal_tolerance"] = float(e.robot_goal.tolerance)
    robot = make_robot(r, **common, **params)
    required = {field.name for field in fields(robot) if field.kw_only}-common.keys()-r.keys()
    _required_section(OmegaConf.create(params), required, label)
    extra = params.keys()-required
    if extra:
        raise ValueError(f"{label} has unexpected parameters: {', '.join(sorted(extra))}")
    return robot


def make_controller(config):
    """Bind a configured robot, environment period and common solver settings."""
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
    robot = _build_robot(config)
    values = _required_section(config.controller.solver,
                               [field.name for field in fields(SolverConfig)], "controller.solver")
    solver = SolverConfig(**values)
    return partial(generate_action, robot, dt=dt, solver=solver, slack_weight=weight)
