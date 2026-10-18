"""Behavioral checks for robot-first crowd updates in CRAX/MJX."""
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_config
from env.jax_social_nav import fixed_order_policy_obs_from_state, reset_env




if importlib.util.find_spec("crax") is None:
    pytest.skip("Optional CRAX environment is not installed", allow_module_level=True)


def make_env(overrides):
    config = make_config(robot="point", extra=["env=crax_point_crowd", *overrides])
    from env.crax_point_crowd import CraxPointCrowd
    return CraxPointCrowd(config)


def scene(env):
    crowd = reset_env(env.nav_cfg, jax.random.PRNGKey(31), 1)
    positions = jnp.zeros_like(crowd.human_positions)
    positions = positions.at[0, 0].set(jnp.array([.9, .35], jnp.float32))
    positions = positions.at[0, 2].set(jnp.array([-3., 3.], jnp.float32))
    goals = positions.at[0, 0].set(jnp.array([-5., .35], jnp.float32))
    # Slot 1 stays at the robot center but is masked out of physics and metrics.
    mask = jnp.zeros_like(crowd.human_mask).at[0, jnp.array([0, 2])].set(1.)
    return crowd._replace(robot_state=jnp.zeros((1, 3), jnp.float32),
        robot_vel=jnp.zeros((1, 2), jnp.float32), robot_action=jnp.zeros((1, 2), jnp.float32),
        goal_pos=jnp.array([[5., 0.]], jnp.float32), prev_dist_to_goal=jnp.array([5.], jnp.float32),
        human_positions=positions, human_goals=goals, human_vels=jnp.zeros_like(positions),
        human_vmaxs=jnp.ones_like(crowd.human_vmaxs), human_mask=mask,
        human_avoid_robot_mask=mask)


@pytest.mark.parametrize("stochastic", [False, True], ids=["deterministic", "gmm_goals"])
def test_obstacles_wait_for_actual_robot_endpoint(stochastic, monkeypatch):
    flag = str(stochastic).lower()
    env = make_env(["env.social_force.avoid_robot=true",
        "+env.social_force.avoid_robot_ratio=1.0", f"env.humans.use_gmm={flag}",
        f"env.humans.end_goal_changing={flag}", f"env.humans.random_goal_changing={flag}",
        "env.humans.goal_change_chance=1.0"])
    before = scene(env)
    key = jax.random.PRNGKey(19)
    snapshots = []
    command = np.array([1., .3], np.float32)
    initial = jax.jit(env.from_crowd)(before, key)
    original_step = env.pipeline_step

    def physics_step(physics, action):
        jax.debug.callback(lambda positions: snapshots.append(np.asarray(positions).copy()),
            physics.mocap_pos[env.crowd_mocap_ids, :2], ordered=True)
        return original_step(physics, action)

    monkeypatch.setattr(env, "pipeline_step", physics_step)
    result, _, _, info = jax.jit(env.step)(initial, jnp.asarray(command))
    jax.block_until_ready(result)
    jax.effects_barrier()
    after = result.crowd
    frame = env.observe(result)
    obstacle_positions = np.asarray(result.physics.mocap_pos[env.crowd_mocap_ids, :2])
    elapsed = float(result.physics.time - initial.physics.time)

    active = np.asarray(before.human_mask[0]) > .5
    assert len(snapshots) == env.substeps
    np.testing.assert_allclose(np.asarray(snapshots)[:, active],
        np.broadcast_to(np.asarray(before.human_positions[0])[active],
                        (env.substeps, active.sum(), 2)), atol=1e-7)
    assert float(jnp.linalg.norm(after.robot_state - before.robot_state)) > .01
    assert float(info["min_clearance"]) > 0.  # Masked slot at the origin is ignored.
    np.testing.assert_allclose(elapsed, env.nav_cfg.dt, atol=1e-6)
    motion_key = jax.random.split(key)[1]
    robot_updated = before._replace(robot_state=after.robot_state,
        robot_vel=after.robot_vel, robot_action=after.robot_action)
    expected = env.kernels["advance"](robot_updated, motion_key)
    old_order = env.kernels["advance"](before, motion_key)
    assert np.max(np.abs(np.asarray(expected.human_vels - old_order.human_vels))) > 1e-5
    for name in ("human_positions", "human_vels", "human_goals"):
        np.testing.assert_allclose(getattr(after, name), getattr(expected, name), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(obstacle_positions[active], np.asarray(after.human_positions[0])[active], atol=1e-6)
    np.testing.assert_allclose(frame, fixed_order_policy_obs_from_state(env.nav_cfg, after)[0], atol=1e-6)
    np.testing.assert_array_equal(after.human_positions[:, 1], before.human_positions[:, 1])
    if stochastic:
        assert not np.array_equal(after.human_goals[:, 2], before.human_goals[:, 2])


def test_contacts_include_robot_phase_and_final_obstacle_update():
    env = make_env(["env.humans.use_gmm=false", "env.humans.end_goal_changing=false",
        "env.social_force.avoid_robot=false"])
    base = scene(env)
    base = base._replace(human_mask=jnp.zeros_like(base.human_mask).at[0, 0].set(1.),
        human_vmaxs=jnp.zeros_like(base.human_vmaxs).at[0, 0].set(3.))
    for distance, speed, final_contact in [(.72, -3., True), (.4, 3., False)]:
        before = base._replace(
            human_positions=base.human_positions.at[0, 0].set(jnp.array([distance, 0.], jnp.float32)),
            human_vels=jnp.zeros_like(base.human_vels).at[0, 0, 0].set(speed),
            human_goals=base.human_goals.at[0, 0].set(jnp.array([np.sign(speed)*5., 0.], jnp.float32)))
        initial = jax.jit(env.from_crowd)(before, jax.random.PRNGKey(19))
        result, _, terminated, info = jax.jit(env.step)(initial, jnp.zeros(2, jnp.float32))
        after = result.crowd
        final_clearance = float(env._clearance(result.physics, after))
        contacted = bool(env._contact(result.physics))
        cost = float(info["cost"])
        elapsed = float(result.physics.time - initial.physics.time)
        assert bool(info["collision"]) and bool(info["physical_contact"]) and bool(terminated)
        assert float(cost) == 1.
        assert contacted == final_contact
        assert float(info["min_clearance"]) < 0.
        np.testing.assert_allclose(elapsed, env.nav_cfg.dt, atol=1e-6)
        if final_contact:
            # New crowd contacts are observed without applying their impulse this step.
            np.testing.assert_allclose(after.robot_state, before.robot_state, atol=1e-6)
            np.testing.assert_allclose(after.robot_vel, before.robot_vel, atol=1e-6)
            np.testing.assert_allclose(info["min_clearance"], final_clearance, atol=1e-6)
        else:
            assert final_clearance > 0.
