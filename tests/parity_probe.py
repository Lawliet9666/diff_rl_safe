"""Compare the committed source and cleaned implementation in separate processes."""
import os
import pickle
from pathlib import Path
import sys

if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'
root, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('add', lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver('mul', lambda *v: np.prod(v).item(), replace=True)
from model import get_jax_model_module
from env.jax_social_nav import config_from_omegaconf, reset_env, fixed_order_policy_obs_from_state, step_env

methods = ['barriernet_gnn', 'barriernet_mlp', 'diff_cvar_gnn', 'diff_cvar_mlp', 'ppo_gnn', 'ppo_mlp']
collected = {}
def record(prefix, tree):
    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, value in paths:
        collected[prefix + '/' + jax.tree_util.keystr(path)] = np.asarray(value)

for robot in ['unicycle', 'single_integrator']:
    for method in methods:
        with initialize_config_dir(version_base=None, config_dir=str(root/'config')):
            cfg = compose(config_name='config', overrides=[f'model={method}', f'robot={robot}', 'device=cpu'])
        module = get_jax_model_module(cfg.model)
        hp = module.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)
        high = np.array([cfg.robot.v_max, cfg.robot.omega_max if robot == 'unicycle' else cfg.robot.v_max], np.float32)
        params = module.init_params_from_config(jax.random.PRNGKey(37), cfg.model, cfg.model.critic, 2, -high, high)
        prefix = f'{robot}/{method}'
        record(prefix+'/params', params)
        ec = config_from_omegaconf(cfg)
        state = reset_env(ec, jax.random.PRNGKey(19), 1)
        # Keep several masked obstacle slots to exercise stable IDs and top-k handling.
        state = state._replace(human_mask=state.human_mask.at[:, 7::3].set(0.0))
        frame = fixed_order_policy_obs_from_state(ec, state)
        history = jnp.repeat(frame[:, None, :], hp.history_len, axis=1)
        has_qp = hasattr(module, 'build_qp_context')
        def forward(p, obs):
            kwargs = {'return_diagnostics': True} if has_qp else {}
            return module.get_action_and_value(p, hp, obs, noise=jnp.zeros((1,2), jnp.float32), **kwargs)
        action_fn = jax.jit(forward)
        step_fn = jax.jit(lambda s, a, k: step_env(ec, s, a, k))
        for i in range(3):
            obs = history.reshape((1,-1))
            if has_qp: record(prefix+f'/context/{i}', module.build_qp_context(obs,hp))
            result = action_fn(params, obs)
            record(prefix+f'/policy/{i}', result)
            state, reward, done, metrics = step_fn(state, result[0], jax.random.PRNGKey(23+i))
            record(prefix+f'/transition/{i}', (state,reward,done,metrics))
            frame = fixed_order_policy_obs_from_state(ec,state)
            history = jnp.concatenate([history[:,1:], frame[:,None,:]], axis=1)
        print(f'Collected {prefix}', flush=True)
        jax.clear_caches()
with output.open('wb') as handle: pickle.dump(collected,handle)
print(f'Saved {len(collected)} arrays to {output}',flush=True)
