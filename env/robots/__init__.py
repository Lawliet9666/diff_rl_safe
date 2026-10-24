"""Select a Robot class; Hydra supplies the complete configuration."""
from .base import Robot, RobotStep, angle_normalize
from .single_integrator import SingleIntegratorRobot
from .double_integrator import DoubleIntegratorRobot
from .unicycle import UnicycleRobot
from .quadrotor import QuadrotorRobot
from .humanoid import HumanoidRobot


def make_robot(config):
    """Dispatch by robot type; each model parses its own configuration block."""
    name = str(config.robot.type).strip().lower()
    if name == "single_integrator":
        model = SingleIntegratorRobot
    elif name == "double_integrator":
        model = DoubleIntegratorRobot
    elif name == "unicycle":
        model = UnicycleRobot
    elif name == "quadrotor":
        model = QuadrotorRobot
    elif name == "humanoid":
        model = HumanoidRobot
    else:
        raise ValueError(f"unsupported robot type {config.robot.type!r}; expected single_integrator, "
                         "double_integrator, unicycle, quadrotor or humanoid")
    return model.from_config(config)


__all__ = ['Robot', 'RobotStep', 'SingleIntegratorRobot', 'DoubleIntegratorRobot', 'UnicycleRobot',
           'QuadrotorRobot', 'HumanoidRobot', 'make_robot', 'angle_normalize']
