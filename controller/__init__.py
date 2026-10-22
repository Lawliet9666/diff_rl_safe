"""Robot-independent CBF-QP action generation."""
from functools import partial

from .cbf_qp import Obstacles, ControlResult, generate_action, UnicycleConfig, QuadrotorConfig, HumanoidConfig

__all__ = ["Obstacles", "ControlResult", "generate_action", "UnicycleConfig", "QuadrotorConfig", "HumanoidConfig", "make_controller"]


def make_controller(config):
    """Select an action generator; preserve the robot/controller call signature."""
    name = config.controller.type
    if name == "cbf_qp":
        return generate_action
    if name == "cbf_qp_slack":
        from .cbf_qp_slack import generate_action as slack_action, validate_slack_weight
        if config.robot.type != "unicycle":
            raise ValueError("cbf_qp_slack currently supports unicycle only")
        weight = config.controller.slack_weight
        validate_slack_weight(weight)
        return partial(slack_action, slack_weight=weight)
    raise ValueError(f"unsupported controller type: {name!r}")
