import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import METHODS, make_config
from env.jax_social_nav import config_from_omegaconf, reset_env, fixed_order_policy_obs_from_state
from model import get_jax_model_module
from trainer.jax_checkpoint import save_jax_checkpoint, load_jax_checkpoint

def initialize(cfg):
    module = get_jax_model_module(cfg.model)
    hp = module.hparams_from_config(cfg.model, cfg.model.critic, 2)
    high = np.array([cfg.robot.v_max, cfg.robot.omega_max if cfg.robot.type == 'unicycle' else cfg.robot.v_max], np.float32)
    params = module.init_params_from_config(jax.random.PRNGKey(0), cfg.model, cfg.model.critic, 2, -high, high)
    ec = config_from_omegaconf(cfg)
    state = reset_env(ec, jax.random.PRNGKey(1), 2)
    frame = fixed_order_policy_obs_from_state(ec, state)
    obs = jnp.repeat(frame[:,None,:], hp.history_len, axis=1).reshape((2,-1))
    return module, hp, params, obs

@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('robot', ['unicycle', 'single_integrator'])
def test_policy_and_value(method, robot):
    module,hp,params,obs = initialize(make_config(method,robot))
    result = jax.jit(lambda p,o: module.get_action_and_value(p,hp,o,noise=jnp.zeros((2,2))))(params,obs)
    for value in result: assert np.all(np.isfinite(value))
    assert np.all(result[0] >= params['action_low']) and np.all(result[0] <= params['action_high'])
    if hasattr(module, 'build_qp_context'):
        frames = obs.reshape((2,hp.history_len,hp.obs_frame_dim))
        blocks = frames[:,:,6:].reshape((2,hp.history_len,hp.num_humans,6))
        blocks = blocks.at[:,:,:,5].set(0).at[:,:,1,5].set(1).at[:,:,4,5].set(1)
        altered = frames.at[:,:,6:].set(blocks.reshape((2,hp.history_len,-1))).reshape(obs.shape)
        selected = np.asarray(module.build_qp_context(altered,hp)['qp_topk_ids'])
        assert all(set(ids[:2]) == {1,4} for ids in selected)
        zero_mask = frames.at[:,:,6:].set(blocks.at[:,:,:,5].set(0).reshape((2,hp.history_len,-1))).reshape(obs.shape)
        assert np.all(np.isfinite(module.policy_action_mean(params,hp,zero_mask)))

def test_topk_tracks_changed_distances():
    module,hp,_params,obs = initialize(make_config())
    frames = obs.reshape((2,hp.history_len,hp.obs_frame_dim))
    blocks = frames[:,:,6:].reshape((2,hp.history_len,hp.num_humans,6))
    blocks = blocks.at[:,:,:,5].set(1).at[:,:,:,0].set(jnp.arange(1,7)).at[:,:,:,1].set(0)
    def context(b):
        return np.asarray(module.build_qp_context(frames.at[:,:,6:].set(b.reshape((2,hp.history_len,-1))).reshape(obs.shape),hp)['qp_topk_ids'])
    assert 5 not in context(blocks)[0]
    assert context(blocks.at[:,:,5,0].set(0.1))[0,0] == 5

@pytest.mark.parametrize('method',['barriernet_gnn','barriernet_mlp'])
def test_gradient_reaches_learned_alpha(method):
    module,hp,params,obs = initialize(make_config(method,'single_integrator'))
    # Approaching obstacle activates the CBF constraint; distant slots are masked.
    frame = jnp.zeros((2,hp.obs_frame_dim),jnp.float32).at[:,5].set(0.3)
    frame = frame.at[:,6:12].set(jnp.array([1.5,0.,1.,0.,0.4,1.],jnp.float32))
    obs = jnp.repeat(frame[:,None,:],hp.history_len,axis=1).reshape(obs.shape)
    gradients = jax.jit(jax.grad(lambda p: jnp.sum(module.policy_action_mean(p,hp,obs))))(params)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(value)) for value in leaves)
    assert np.linalg.norm(gradients['actor']['human_alpha_head']['bias']) > 1e-7

def test_qp_backward_against_active_constraint_solution():
    from solver.jax_qpth import QPFunction
    # min ||u||² / 2, subject to u_x >= threshold has u_x = threshold.
    # This well-conditioned fixture has an independently known derivative.
    qp = QPFunction(eps=1e-12,maxIter=40,notImprovedLim=3,check_Q_spd=False)
    q = jnp.eye(2,dtype=jnp.float64)
    p = jnp.zeros(2,jnp.float64)
    g = jnp.array([[-1.,0.]],jnp.float64)
    empty = jnp.empty(0,jnp.float64)
    def solve(threshold):
        return qp(q,p,g,jnp.array([-threshold]),empty,empty)[0,0]
    derivative = jax.grad(solve)(jnp.float64(0.7))
    finite_difference = (solve(jnp.float64(0.701))-solve(jnp.float64(0.699)))/0.002
    np.testing.assert_allclose(solve(jnp.float64(0.7)),0.7,atol=1e-8)
    np.testing.assert_allclose(derivative,1.,atol=1e-6)
    np.testing.assert_allclose(derivative,finite_difference,atol=1e-6)

def test_checkpoint_roundtrip(tmp_path):
    _module,_hp,params,_obs = initialize(make_config())
    path = save_jax_checkpoint(tmp_path,params,step=128,performance=0.0)
    payload = load_jax_checkpoint(path)
    assert payload['step'] == 128
    for actual,expected in zip(jax.tree_util.tree_leaves(payload['params']),jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(actual,expected)

def test_evaluation_with_all_obstacles_masked(monkeypatch):
    from trainer import jax_ppo_eval
    cfg = make_config('ppo_mlp')
    module,hp,params,_obs = initialize(cfg)
    ec = config_from_omegaconf(cfg)
    state = reset_env(ec,jax.random.PRNGKey(1),1)
    state = state._replace(human_mask=jnp.zeros_like(state.human_mask))
    monkeypatch.setattr(jax_ppo_eval,'reset_env_with_keys',lambda *_:state)
    metrics = jax_ppo_eval.evaluate_jax_env(
        params,hp,ec,episodes=1,seeds=[100],
        policy_action_mean_fn=module.policy_action_mean,
        policy_action_to_env_action_fn=module.policy_action_to_env_action)
    assert metrics['mean_min_clearance'] is None
    assert all(value is None or np.isfinite(value) for value in metrics.values())
