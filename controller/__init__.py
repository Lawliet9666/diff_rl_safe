"""Robot-independent CBF-QP action generation."""
from dataclasses import fields
from functools import partial

from omegaconf import OmegaConf

from env.robots.base import Obstacles

from .cbf_qp import (
    ControlResult, generate_action, validate_slack_weight,
    SolverConfig, UnicycleConfig, QuadrotorConfig, HumanoidConfig,
)

__all__ = ["Obstacles", "ControlResult", "generate_action", "UnicycleConfig", "QuadrotorConfig", "HumanoidConfig", "make_controller"]


def _build_controller_config(config):
    r, c, e = config.robot, config.controller, config.env
    if r.type not in {"unicycle", "quadrotor", "humanoid"}:
        raise ValueError("controller robot must be unicycle, quadrotor or humanoid")

    def required_section(node, names, label):
        values = OmegaConf.to_container(node, resolve=True)
        missing = set(names)-values.keys()
        if missing:
            raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
        return values

    def robot_section(node, names, label):
        groups = ("nominal", "cbf", "qp")
        sections = required_section(node, groups, label)
        params = {key: value for key, value in sections.items() if key not in groups}
        blocks = [sections["nominal"], sections["qp"]]
        if r.type == "humanoid":
            cbf = required_section(OmegaConf.create(sections["cbf"]),
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
        return required_section(OmegaConf.create(params), names, label)

    solver = SolverConfig(**required_section(c.solver, [f.name for f in fields(SolverConfig)],
                                            "controller.solver"))
    common = dict(radius=float(r.radius), margin=float(e.safety_margin),
                  omega_max=float(r.omega_max), solver=solver)
    if r.type == "unicycle":
        names = {f.name for f in fields(UnicycleConfig)}-common.keys()-{"v_max", "goal_tolerance"}
        params = robot_section(c.unicycle, names, "controller.unicycle")
        params["weights"] = tuple(params["weights"])
        return UnicycleConfig(goal_tolerance=float(e.robot_goal.tolerance),
                              v_max=float(r.v_max), **common, **params)
    elif r.type == "humanoid":
        common.pop("omega_max")  # Physical limits belong to the robot model.
        names = {f.name for f in fields(HumanoidConfig)}-common.keys()-{"dt", "goal_tolerance"}
        params = robot_section(c.humanoid, names, "controller.humanoid")
        for name in ("gains", "weights"):
            params[name] = tuple(params[name])
        return HumanoidConfig(dt=float(e.dt), goal_tolerance=float(e.robot_goal.tolerance),
                              **common, **params)
    else:
        physical = {name: float(r[name]) for name in
                    ["mass", "gravity", "thrust_rate_max", "thrust_min", "thrust_max",
                     "tilt_max", "altitude", "altitude_min", "altitude_max"]}
        names = {f.name for f in fields(QuadrotorConfig)}-common.keys()-physical.keys()
        params = robot_section(c.quadrotor, names, "controller.quadrotor")
        for name in ("gains", "weights", "acceleration_limits"):
            params[name] = tuple(params[name])
        return QuadrotorConfig(**common, **physical, **params)


def make_controller(config):
    """Build and validate parameters; return a ready-to-use CBF-QP filter."""
    if config.controller.type != "cbf_qp":
        raise ValueError(f"unsupported controller type: {config.controller.type!r}")
    weight = config.controller.slack_weight
    validate_slack_weight(weight)
    if weight > 0 and config.robot.type != "unicycle":
        raise ValueError("positive slack_weight currently supports unicycle only")
    params = _build_controller_config(config)
    return partial(generate_action, control_config=params, slack_weight=weight)
