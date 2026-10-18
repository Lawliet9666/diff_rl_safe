"""Learned BarrierNet distance heads, QP gradients and legacy checkpoint behavior."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import open_dict

from conftest import make_config
from test_methods import initialize
from trainer.jax_checkpoint import load_jax_checkpoint, save_jax_checkpoint


METHODS = ["barriernet_gnn", "barriernet_mlp"]


def _observation(hp, active_only=False):
    frame = jnp.zeros((2, hp.obs_frame_dim), jnp.float32).at[:, 5].set(.3)
    blocks = jnp.zeros((2, hp.num_humans, 6), jnp.float32)
    blocks = blocks.at[:, :, 0].set(jnp.arange(1, hp.num_humans + 1))
    blocks = blocks.at[:, :, 4].set(.4).at[:, :, 5].set(1.)
    if active_only:
        blocks = blocks.at[:, :, 5].set(0.).at[:, 0, 5].set(1.)
        # Keep the active constraint inside actuator bounds, so its gradient is observable.
        blocks = blocks.at[:, 0, 0].set(1.5).at[:, 0, 2].set(1.)
    frame = frame.at[:, 6:].set(blocks.reshape(2, -1))
    return jnp.repeat(frame[:, None], hp.history_len, axis=1).reshape(2, -1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("robot", ["unicycle", "single_integrator"])
def test_distance_head_bounds_and_changed_nearest_obstacles(method, robot):
    module, hp, params, _ = initialize(make_config(method, robot))
    assert hp.learn_r_safe
    obs = _observation(hp)

    @jax.jit
    def diagnostics(p, o):
        return module.get_action_and_value(p, hp, o, noise=jnp.zeros((2, 2)),
                                          return_diagnostics=True)[-1]

    initial = diagnostics(params, obs)["r_safe"]
    assert initial.shape == (2, hp.qp_obs_top_k)
    assert np.all(np.asarray(initial) > .75)
    assert np.all(np.asarray(initial) < 1.875)
    for logit, expected in [(-80., .75), (80., 1.875)]:
        modified = deepcopy(params)
        head = modified["actor"]["human_rsafe_head"]
        head["weight"] = jnp.zeros_like(head["weight"])
        head["bias"] = jnp.full_like(head["bias"], logit)
        np.testing.assert_allclose(diagnostics(modified, obs)["r_safe"], expected, atol=1e-6)

    context = module.build_qp_context(obs, hp)
    assert 5 not in np.asarray(context["qp_topk_ids"])[0]
    frames = obs.reshape(2, hp.history_len, hp.obs_frame_dim)
    blocks = frames[:, :, 6:].reshape(2, hp.history_len, hp.num_humans, 6)
    blocks = blocks.at[:, :, 5, 0].set(.8)
    changed = frames.at[:, :, 6:].set(blocks.reshape(2, hp.history_len, -1)).reshape(2, -1)
    assert np.asarray(module.build_qp_context(changed, hp)["qp_topk_ids"])[0, 0] == 5
    assert np.all(np.isfinite(diagnostics(params, changed)["r_safe"]))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("robot", ["unicycle", "single_integrator"])
def test_qp_gradients_reach_alpha_and_distance_with_masks(method, robot):
    module, hp, params, _ = initialize(make_config(method, robot))
    obs = _observation(hp, active_only=True)
    gradient = jax.jit(jax.grad(lambda p: module.policy_action_mean(p, hp, obs).sum()))(params)
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient))
    for name in ("human_alpha_head", "human_rsafe_head"):
        assert np.linalg.norm(gradient["actor"][name]["bias"]) > 1e-7
    if method.endswith("mlp"):
        # Masked slots have independent MLP outputs and contribute no QP gradient.
        np.testing.assert_allclose(gradient["actor"]["human_rsafe_head"]["bias"][1:], 0., atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_old_config_checkpoint_keeps_fixed_distance(method, tmp_path):
    cfg = make_config(method, extra=["model.actor.learn_r_safe=false"])
    module, hp, params, obs = initialize(cfg)
    assert "human_rsafe_head" not in params["actor"]
    path = save_jax_checkpoint(tmp_path, params, step=128, performance=0.)
    payload = load_jax_checkpoint(path)

    # Historical resolved configs lack both new fields.
    with open_dict(cfg.model.actor):
        del cfg.model.actor["learn_r_safe"]
        del cfg.model.actor["margin_extra_max"]
    legacy_hp = module.hparams_from_config(cfg.model, cfg.model.critic, 2)
    assert not legacy_hp.learn_r_safe
    old_result = module.get_action_and_value(params, hp, obs, noise=jnp.zeros((2, 2)),
                                             return_diagnostics=True)
    loaded_result = module.get_action_and_value(payload["params"], legacy_hp, obs,
                                                noise=jnp.zeros((2, 2)), return_diagnostics=True)
    for actual, expected in zip(jax.tree_util.tree_leaves(loaded_result),
                                jax.tree_util.tree_leaves(old_result)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(loaded_result[-1]["r_safe"], .75, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [-.1, 0., float("nan"), float("inf")])
def test_invalid_learned_distance_range_is_rejected(method, value):
    cfg = make_config(method)
    cfg.model.actor.margin_extra_max = value
    from model import get_jax_model_module
    with pytest.raises(ValueError, match="margin_extra_max"):
        get_jax_model_module(cfg.model).hparams_from_config(cfg.model, cfg.model.critic, 2)
