"""CBF-QP controller factory."""
from dataclasses import fields
from functools import partial
import math

from omegaconf import OmegaConf

from env.robots import make_robot
from env.robots.base import Obstacles
from .cbf_qp import generate_action, ControlResult, SolverConfig, validate_slack_weight

__all__ = ["Obstacles", "generate_action", "ControlResult", "SolverConfig",
           "make_controller", "make_safety_filter"]


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
    dt = float(config.env.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("env.dt must be positive")
    if config.robot.type not in {"single_integrator", "unicycle", "double_integrator", "quadrotor", "humanoid"}:
        raise ValueError("controller robot must be single_integrator, unicycle, double_integrator, quadrotor or humanoid")
    robot = make_robot(config)
    values = _required_section(config.controller.solver,
                               [field.name for field in fields(SolverConfig)], "controller.solver")
    solver = SolverConfig(**values)
    action_fn = partial(generate_action, robot, dt=dt, backend="osqp",
                        solver=solver, slack_weight=weight, qp_obs_top_k=top_k)
    return robot, solver, action_fn


def make_safety_filter(config, env_cfg):
    """Build only an explicitly enabled, ordinary-PPO JAX execution filter."""
    import jax

    section = config.trainer.safety_filter
    if not isinstance(section.enabled, bool):
        raise ValueError("trainer.safety_filter.enabled must be true or false")
    physical = config.trainer.physical_limits
    if not isinstance(physical.enabled, bool):
        raise ValueError("trainer.physical_limits.enabled must be true or false")
    if section.enabled and physical.enabled:
        raise ValueError("enable either physical_limits or safety_filter, not both")
    if not (section.enabled or physical.enabled):
        return None
    if config.model.type not in {"ppo_mlp", "ppo_gnn"}:
        raise ValueError("external safety filter requires PPO-MLP or PPO-GNN; policy QP layers remain separate")
    from env.jax_env.jax_social_nav import JaxSocialNavConfig
    if not isinstance(env_cfg, JaxSocialNavConfig):
        raise ValueError("external safety filter currently requires the JAX environment")
    if not jax.config.x64_enabled:
        raise ValueError("external safety filter requires JAX_ENABLE_X64=true")
    if physical.enabled:
        return partial(generate_action, env_cfg.robot, dt=env_cfg.dt, backend="jax",
                       constraint_mode="physical")
    if config.robot.type == 'humanoid' and env_cfg.robot.barrier != 'distance':
        raise ValueError("SF-PPO humanoid requires the distance HOCBF")
    top_k = section.qp_obs_top_k
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError("trainer.safety_filter.qp_obs_top_k must be null or a positive integer")
    for name, minimum in (("max_iter", 1), ("speed_facets", 4)):
        value = section[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"trainer.safety_filter.{name} must be an integer >= {minimum}")
    for name in ('eps', 'slack_weight'):
        value = section[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 or (name != 'slack_weight' and value == 0):
            raise ValueError(f"invalid trainer.safety_filter.{name}")
    return partial(generate_action, env_cfg.robot, dt=env_cfg.dt, backend="jax",
        qp_obs_top_k=min(top_k, env_cfg.num_humans) if top_k is not None else env_cfg.num_humans,
        slack_weight=float(section.slack_weight), speed_facets=section.speed_facets,
        max_iter=section.max_iter, eps=float(section.eps))
