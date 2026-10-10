import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['MPLBACKEND'] = 'Agg'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('add', lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver('mul', lambda *v: math.prod(v), replace=True)

METHODS = ['barriernet_gnn', 'barriernet_mlp', 'diff_cvar_gnn', 'diff_cvar_mlp', 'ppo_gnn', 'ppo_mlp']

def make_config(method='barriernet_gnn', robot='unicycle', extra=()):
    overrides = [f'model={method}', f'robot={robot}', 'device=cpu',
                 'env.humans.num_humans=6', 'env.max_steps=8', 'model.history_len=2',
                 'model.actor.hidden_dim=16', 'model.actor.control_hidden_dim=16',
                 'model.critic.hidden_dim=16', 'model.critic.hidden_dim2=16']
    if method.endswith('gnn'):
        overrides += ['model.actor.graph_hidden_dim=8', 'model.actor.graph_latent_dim=4', 'model.actor.graph_layers=1']
    else:
        overrides += ['model.actor.encoder_hidden_dim=8', 'model.actor.encoder_latent_dim=4', 'model.actor.encoder_layers=1']
    if not method.startswith('ppo'):
        overrides += ['model.actor.scalar_hidden_dim=16']
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'config')):
        return compose(config_name='config', overrides=overrides+list(extra))

@pytest.fixture(autouse=True)
def release_compilations():
    yield
    import jax
    jax.clear_caches()
