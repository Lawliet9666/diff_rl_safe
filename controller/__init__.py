"""Robot-independent CBF-QP action generation."""
from functools import partial

from .cbf_qp import (
    Obstacles, ControlResult, generate_action, validate_slack_weight,
    UnicycleConfig, QuadrotorConfig, HumanoidConfig,
)

__all__ = ["Obstacles", "ControlResult", "generate_action", "UnicycleConfig", "QuadrotorConfig", "HumanoidConfig", "make_controller"]


def make_controller(config):
    """Select a hard (zero weight) or relaxed (positive weight) CBF-QP."""
    if config.controller.type != "cbf_qp":
        raise ValueError(f"unsupported controller type: {config.controller.type!r}")
    weight = config.controller.slack_weight
    validate_slack_weight(weight)
    if weight > 0 and config.robot.type != "unicycle":
        raise ValueError("positive slack_weight currently supports unicycle only")
    return partial(generate_action, slack_weight=weight)
