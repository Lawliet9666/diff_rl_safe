"""Robot selection and the existing planar SocialNav stepping interface."""
from dataclasses import fields

from .base import Robot, RobotStep, angle_normalize
from .single_integrator import SingleIntegratorRobot
from .unicycle import UnicycleRobot
from .quadrotor import QuadrotorRobot
from .humanoid import HumanoidRobot


_MODELS = dict(single_integrator=SingleIntegratorRobot, unicycle=UnicycleRobot,
               quadrotor=QuadrotorRobot, humanoid=HumanoidRobot)


def make_robot(robot_type, config=None, **parameters):
    """Select a model by name, using explicit physical parameters or a config.

    A supplied config must contain every original physical field. Keyword-only
    settings may use robot defaults or explicit overrides. No controller is imported.
    """
    if config is None and not isinstance(robot_type, str):
        config = robot_type
        robot_type = config.type
    name = str(robot_type).strip().lower()
    if name not in _MODELS:
        raise ValueError(f"unsupported robot type {robot_type!r}; expected one of {tuple(_MODELS)}")
    model = _MODELS[name]
    if config is not None:
        configured = {field.name: getattr(config, field.name) for field in fields(model)
                      if not field.kw_only or hasattr(config, field.name)}
        parameters = {**configured, **parameters}
    for field in fields(model):
        if field.name in parameters and isinstance(field.default, tuple):
            parameters[field.name] = tuple(parameters[field.name])
    return model(**parameters)


def robot_step(cfg, state, action):
    """Keep the original SocialNav tuple and its planar integration behavior."""
    if cfg.robot_type not in {"single_integrator", "unicycle"}:
        raise ValueError("planar robot_step requires single_integrator or unicycle")
    return tuple(cfg.robot.step_with_info(state, action, cfg.dt))


__all__ = ['Robot', 'RobotStep', 'SingleIntegratorRobot', 'UnicycleRobot',
           'QuadrotorRobot', 'HumanoidRobot', 'make_robot', 'robot_step', 'angle_normalize']
