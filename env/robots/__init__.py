"""Robot selection and the existing planar SocialNav stepping interface."""
from dataclasses import fields

from .base import Robot, RobotStep, angle_normalize
from .single_integrator import SingleIntegratorRobot
from .unicycle import UnicycleRobot
from .quadrotor import QuadrotorRobot


_MODELS = dict(single_integrator=SingleIntegratorRobot, unicycle=UnicycleRobot,
               quadrotor=QuadrotorRobot)


def make_robot(robot_type, config=None, **parameters):
    """Select a model by name, using explicit physical parameters or a config.

    A supplied config must contain every physical field of the selected model.
    Extra controller settings are ignored; no controller module is imported.
    """
    if config is None and not isinstance(robot_type, str):
        config = robot_type
        robot_type = config.type
    name = str(robot_type).strip().lower()
    if name not in _MODELS:
        raise ValueError(f"unsupported robot type {robot_type!r}; expected one of {tuple(_MODELS)}")
    model = _MODELS[name]
    if config is not None:
        parameters = {**{field.name: getattr(config, field.name) for field in fields(model)},
                      **parameters}
    return model(**parameters)


def robot_step(cfg, state, action):
    """Keep the original SocialNav tuple and its planar integration behavior."""
    parameters = dict(v_max=cfg.robot_vmax)
    if cfg.robot_type == "unicycle":
        parameters['omega_max'] = cfg.robot_omega_max
    robot = make_robot(cfg.robot_type, **parameters)
    return tuple(robot.step_with_info(state, action, cfg.dt))


__all__ = ['Robot', 'RobotStep', 'SingleIntegratorRobot', 'UnicycleRobot',
           'QuadrotorRobot', 'make_robot', 'robot_step', 'angle_normalize']
