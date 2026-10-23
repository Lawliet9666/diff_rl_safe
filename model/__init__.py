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
    """Adapt deterministic inference to action_fn(observation) -> (action, info).

    Action scaling and optional QP context stay in the existing model functions.
    PPO training still uses get_action_and_value for probability/value terms.
    """
    def generate_action(observation, **kwargs):
        policy_action = policy_action_mean_fn(params, hparams, observation, **kwargs)
        action = policy_action_to_env_action_fn(params, policy_action)
        return action, {"policy_action": policy_action}

    return generate_action


__all__ = ["get_model_module", "make_action_fn"]
