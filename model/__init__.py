"""JAX-only model package and lazy policy selection."""

from importlib import import_module


_MODEL_MODULES = {
    "barriernet_gnn": "model.barriernet_gnn",
    "barriernet_mlp": "model.barriernet_mlp",
    "diff_cvar_gnn": "model.diff_cvar_gnn",
    "diff_cvar_mlp": "model.diff_cvar_mlp",
    "ppo_gnn": "model.ppo_gnn",
    "ppo_mlp": "model.ppo_mlp",
}


def get_model_module(model_cfg):
    model_type = str(model_cfg.get("type") if isinstance(model_cfg, dict) else model_cfg.type)
    module_name = _MODEL_MODULES.get(model_type)
    if module_name is None:
        raise ValueError(f"unsupported model.type={model_type!r}; expected one of {tuple(_MODEL_MODULES)}")
    return import_module(module_name)


def make_action_fn(params, hparams, policy_action_mean_fn, policy_action_to_env_action_fn):
    """Wrap deterministic inference as action_fn(obs) -> (action, info)."""
    def generate_action(observation, **kwargs):
        policy_action = policy_action_mean_fn(params, hparams, observation, **kwargs)
        action = policy_action_to_env_action_fn(params, policy_action)
        return action, {"policy_action": policy_action}

    return generate_action


__all__ = ["get_model_module", "make_action_fn", "prepare_robot_policy"]


def prepare_robot_policy(config):
    """Resolve Robot-derived policy dimensions and validate model support."""
    from omegaconf import open_dict
    from env.robots import make_robot
    robot_type = str(config.robot.type)
    extended = robot_type in {"double_integrator", "humanoid", "quadrotor"}
    if extended:
        if config.model.type not in {"barriernet_gnn", "barriernet_mlp"}:
            raise ValueError("New RL robots require BarrierNet GNN or MLP")
        if config.env.get("backend", "jax") != "jax":
            raise ValueError("New RL robots currently require the JAX environment")
        if robot_type == "humanoid" and config.controller.humanoid.barrier != "distance":
            raise ValueError("Humanoid RL currently requires the distance HOCBF")
    robot = make_robot(config)
    with open_dict(config):
        config.env.act_dim = robot.control_dim
        if config.model.type in {"barriernet_gnn", "barriernet_mlp"} and "num_humans" not in config.model:
            # Infer missing human-slot counts in planar checkpoints.
            config.model.num_humans = (int(config.model.obs_frame_dim)-6)//6
        if extended:
            config.model.obs_frame_dim = 6 + 6*int(config.model.num_humans) + robot.state_dim
    return robot
