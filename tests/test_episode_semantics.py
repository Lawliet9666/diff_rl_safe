"""Regression checks for evaluation priority and configurable timeout targets."""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from conftest import METHODS, make_config
from trainer.jax_episode import episode_outcomes
from trainer.jax_ppo_core import compute_brax_gae
from trainer.jax_ppo_rollout import make_rollout_collector
from trainer.jax_timeout import compute_timeout_gae, timeout_bootstrap_from_config


def test_exclusive_outcomes_at_goal_collision_and_limit():
    # Includes simultaneous goal/collision/limit, goal at limit and no terminal.
    done = jnp.array([1, 1, 1, 1, 0])
    success = jnp.array([1, 1, 0, 0, 1])
    collision = jnp.array([1, 0, 1, 0, 1])
    result = jax.jit(episode_outcomes)(done, success, collision)
    np.testing.assert_array_equal(result[0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(result[1], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(result[2], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(np.asarray(result).sum(axis=0), done)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('enabled', [False, True])
def test_all_methods_wire_the_timeout_switch(method, enabled, monkeypatch):
    from trainer import jax_ppo_base_trainer as trainers

    cfg = make_config(method, extra=[f'trainer.timeout_bootstrap={str(enabled).lower()}'])
    captured = []
    def setup_shell(self, config):
        self.total_timesteps = config.trainer.total_timesteps
        self.timesteps_per_batch = config.trainer.timesteps_per_batch

    monkeypatch.setattr(trainers.JaxTrainerBase, '__init__', setup_shell)
    monkeypatch.setattr(trainers, 'make_rollout_collector',
                        lambda *args, **kwargs: captured.append(kwargs['timeout_aware']))
    trainer = trainers.JaxPPOBaseTrainer(cfg)
    assert trainer.timeout_bootstrap is enabled
    assert captured == [enabled]


def test_defaults_and_legacy_config_compatibility():
    assert timeout_bootstrap_from_config(make_config()) is True
    for backend, expected in [('jax', False), ('crax_mjx', True)]:
        old = OmegaConf.create({'env': {'backend': backend}, 'trainer': {}})
        assert timeout_bootstrap_from_config(old) is expected
        old.trainer.timeout_bootstrap = False
        assert timeout_bootstrap_from_config(old) is False
        old.trainer.timeout_bootstrap = True
        assert timeout_bootstrap_from_config(old) is True
        old.trainer.timeout_bootstrap = 'false'
        with pytest.raises(ValueError, match='true or false'):
            timeout_bootstrap_from_config(old)


class ToyPolicy:
    @staticmethod
    def critic_value(params, hp, obs):
        return obs.sum(axis=-1)

    @staticmethod
    def get_action_and_value(params, hp, obs, **kwargs):
        action = jnp.zeros((obs.shape[0], 2), jnp.float32)
        zero = jnp.zeros(obs.shape[0], jnp.float32)
        return action, zero, zero, ToyPolicy.critic_value(params, hp, obs), action


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('physical_metrics', [False, True])
def test_final_observation_bootstrap_and_legacy_rollback(enabled, physical_metrics):
    def step(cfg, state, action, key):
        zero = jnp.zeros_like(state)
        one = jnp.ones_like(state)
        metrics = dict(success=zero, collision=zero, timeout=one,
                       episode_return=one, episode_length=one, min_clearance=one,
                       boundary_violation=zero, robot_omega_delta=zero,
                       robot_omega_delta_penalty=zero)
        if physical_metrics:
            metrics.update(terminated=zero, truncated=one, tracking_speed_error=one,
                           tracking_omega_error=zero, tracking_lateral_speed=zero,
                           actuator_saturation=zero, physical_contact=zero, cost=zero)
        return state + 1, one, one, metrics

    def reset(cfg, key, n):
        return jnp.full((n,), 100., jnp.float32)

    collect = make_rollout_collector(
        ToyPolicy, False, step_fn=step, reset_fn=reset,
        observe_fn=lambda cfg, state: state[:, None], timeout_aware=enabled)
    state = (jnp.zeros(1, jnp.float32), jnp.zeros((1, 2, 1), jnp.float32),
             jnp.zeros(1, jnp.float32))
    batch, final, _ = jax.jit(lambda: collect(
        {}, SimpleNamespace(act_dim=2), None, state, jax.random.PRNGKey(0), 2))()
    # A reset observation is 100: its critic value 200 must never bootstrap
    # the preceding episode, whose final history is [0, 1].
    np.testing.assert_array_equal(final[1], [[[100.], [100.]]])
    if enabled:
        np.testing.assert_array_equal(batch['next_values'], [[1.], [201.]])
        adv, returns = compute_timeout_gae(batch['rew'], batch['val'],
            batch['terminated'], batch['truncated'], batch['next_values'], .9, .95)
        np.testing.assert_allclose(returns, [1.9, 181.9], rtol=1e-6)
        np.testing.assert_allclose(adv, [1.9, -18.1], rtol=1e-6)
    else:
        assert 'terminated' not in batch and 'next_values' not in batch
        _, returns = compute_brax_gae(batch['rew'], batch['val'], batch['done'],
                                     batch['next_value'], .9, .95)
        np.testing.assert_allclose(returns, [1., 1.])
    assert ('control/tracking_speed_error' in batch) is physical_metrics


def test_success_and_collision_do_not_bootstrap_and_gae_stops_at_reset():
    rewards = jnp.ones((1, 3), jnp.float32)
    _, returns = compute_timeout_gae(rewards, jnp.zeros_like(rewards),
        jnp.array([[1., 1., 0.]]), jnp.array([[0., 0., 1.]]),
        jnp.array([[100., 200., 3.]]), .9, .95)
    np.testing.assert_allclose(returns, [1., 1., 3.7], rtol=1e-6)
    for later_reward in [-1e6, 1e6]:
        _, returns = compute_timeout_gae(jnp.array([[1.], [later_reward]]),
            jnp.zeros((2, 1)), jnp.array([[0.], [1.]]), jnp.array([[1.], [0.]]),
            jnp.array([[3.], [0.]]), .9, .95)
        np.testing.assert_allclose(returns[0], 3.7, rtol=1e-6)


class Physics(NamedTuple):
    qpos: object
    qvel: object
    mocap_pos: object


class Crowd(NamedTuple):
    human_mask: object
    kind: object


class State(NamedTuple):
    physics: Physics
    crowd: Crowd


class ToyPointEnv:
    nav_cfg = SimpleNamespace(max_steps=2, dt=.1)

    @staticmethod
    def reset(key):
        return State(Physics(jnp.zeros(3, jnp.float32), jnp.zeros(3, jnp.float32),
                             jnp.zeros((1, 3), jnp.float32)),
                     Crowd(jnp.ones((1, 1), jnp.float32), key[1]))

    @staticmethod
    def observe(state):
        return jnp.zeros(6, jnp.float32)

    @staticmethod
    def step(state, action):
        kind = state.crowd.kind
        # Raw flags deliberately conflict: limit plus goal and/or collision.
        metrics = dict(success=(kind < 2).astype(jnp.float32),
                       collision=(kind == 0).astype(jnp.float32),
                       timeout=jnp.float32(1), cost=jnp.float32(0),
                       min_clearance=jnp.float32(.2), physical_contact=jnp.float32(0))
        return state, jnp.float32(3), jnp.float32(1), metrics


def test_both_evaluators_agree_on_conflicting_raw_flags(monkeypatch):
    from trainer import crax_eval, jax_ppo_eval

    hp = SimpleNamespace(history_len=2)
    module = SimpleNamespace(
        hparams_from_config=lambda *args, **kwargs: hp,
        policy_action_mean=lambda params, hp, obs: jnp.zeros((obs.shape[0], 2), jnp.float32),
        policy_action_to_env_action=lambda params, action: action)
    monkeypatch.setattr(crax_eval, 'get_jax_model_module', lambda config: module)
    env = ToyPointEnv()
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(3, dtype=jnp.uint32))
    trace, _ = crax_eval._make_evaluation(env, module, hp, False)({}, keys)
    expected = ([0, 1, 0], [1, 0, 0], [0, 0, 1])
    for name, flags in zip(('success', 'collision', 'timeout'), expected):
        np.testing.assert_array_equal(trace[name].sum(axis=0), flags)

    monkeypatch.setattr(jax_ppo_eval, 'reset_env_with_keys',
                        lambda cfg, keys: jax.vmap(env.reset)(keys))
    monkeypatch.setattr(jax_ppo_eval, '_policy_obs_frame_from_state',
                        lambda cfg, state, hp: jax.vmap(env.observe)(state))
    monkeypatch.setattr(jax_ppo_eval, 'step_env',
                        lambda cfg, state, action, key: jax.vmap(env.step)(state, action))
    result = jax_ppo_eval._evaluate_seed_values({}, hp, env.nav_cfg,
        np.arange(3), module.policy_action_mean, module.policy_action_to_env_action)
    for flags, wanted in zip(result[1:4], expected):
        np.testing.assert_array_equal(flags, wanted)
    metrics = crax_eval.evaluate_crax_point(make_config('ppo_mlp'), {}, episodes=1,
                                           seeds=[0, 1, 2], env=env)
    for name in ('success_rate', 'collision_rate', 'timeout_rate'):
        assert metrics[name] == pytest.approx(1 / 3)
    assert metrics['mean_return'] == 3.0
