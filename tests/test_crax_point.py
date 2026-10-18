"""Physics/task integration and native MuJoCo parity for the optional CRAX backend."""
import importlib.util
import numpy as np
import pytest
import jax
import jax.numpy as jnp

if importlib.util.find_spec("crax") is None:
    pytest.skip("Optional CRAX environment is not installed", allow_module_level=True)

from conftest import make_config
from env.jax_social_nav import reset_env, step_env, fixed_order_policy_obs_from_state
from trainer.jax_timeout import compute_timeout_gae


def test_timeout_final_observation_bootstrap():
    adv, ret = compute_timeout_gae(jnp.array([[1.], [10.]]), jnp.array([[1.], [10.]]),
        jnp.array([[0.], [1.]]), jnp.array([[1.], [0.]]), jnp.array([[5.], [20.]]), .99, .98)
    np.testing.assert_allclose(np.asarray(ret), [5.95, 10.], atol=1e-6)
    np.testing.assert_allclose(np.asarray(adv), [4.95, 0.], atol=1e-6)


def test_crax_task_batch_events_tracking_and_native_parity():
    import mujoco
    from mujoco import mjx
    from env.crax_point_crowd import CraxPointCrowd
    config = make_config(robot="point", extra=["env=crax_point_crowd", "env.max_steps=60"])
    env = CraxPointCrowd(config)
    key = jax.random.PRNGKey(7)
    state = jax.jit(env.reset)(key)
    expected_reset = reset_env(env.nav_cfg, jax.random.split(key)[1], 1)
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(state.crowd), jax.tree_util.tree_leaves(expected_reset)):
        if np.issubdtype(actual_leaf.dtype, np.floating):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)
        else:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)
    for value in jax.tree_util.tree_leaves(state.physics):
        if hasattr(value, "dtype") and np.issubdtype(value.dtype, np.floating):
            assert value.dtype == jnp.float32
    np.testing.assert_array_equal(env.observe(state), fixed_order_policy_obs_from_state(env.nav_cfg, state.crowd)[0])
    expected = step_env(env.nav_cfg, state.crowd, jnp.zeros((1, 2), jnp.float32), key)[0]
    actual = env.kernels["advance"](state.crowd, key)
    for field in ("human_positions", "human_vels", "human_goals"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-6)
    crowd = state.crowd._replace(robot_state=jnp.zeros((1, 3), jnp.float32),
        robot_vel=jnp.zeros((1, 2), jnp.float32), robot_action=jnp.zeros((1, 2), jnp.float32),
        goal_pos=jnp.array([[5., 0.]], jnp.float32), prev_dist_to_goal=jnp.array([5.], jnp.float32),
        human_mask=jnp.zeros_like(state.crowd.human_mask), human_vmaxs=jnp.zeros_like(state.crowd.human_vmaxs))
    # Free motion: same model, controller, timestep and solver in native MuJoCo.
    initial = jax.jit(env.from_crowd)(crowd, key)
    native = mujoco.MjData(env.model)
    native.qpos[:], native.qvel[:] = initial.physics.qpos, initial.physics.qvel
    native.mocap_pos[:] = initial.physics.mocap_pos
    mujoco.mj_forward(env.model, native)
    command = np.array([1., .3], np.float32)
    mass = np.zeros((3, 3))
    for _ in range(env.substeps):
        yaw, omega = native.qpos[2], native.qvel[2]
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        acc = np.r_[heading*(command[0]-native.qvel[:2]@heading)/env.tracking_tau,
                    (command[1]-omega)/env.tracking_tau]
        mujoco.mj_fullM(env.model, native, mass)
        force = mass@acc + native.qfrc_bias-native.qfrc_passive
        gears = env.model.actuator_gear[:2, 0]
        desired = np.array([force[:2]@heading/gears[0], force[2]/gears[1]])
        limited = np.clip(desired, env.model.actuator_forcerange[:, 0], env.model.actuator_forcerange[:, 1])
        native.ctrl[:] = np.clip([limited[0], gears[1]*omega+limited[1]/env.model.actuator_gainprm[1, 0]],
            env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(env.model, native)
        mujoco.mj_forward(env.model, native)
    step = jax.jit(env.step)
    # A stationary masked scene isolates source task arithmetic from the
    # intentional swept-clearance/physical-actuation differences.
    stationary, physical_reward, physical_done, _ = step(initial, jnp.zeros(2, jnp.float32))
    legacy, legacy_reward, legacy_done, _ = step_env(env.nav_cfg, crowd,
        jnp.zeros((1, 2), jnp.float32), jax.random.split(key)[1])
    np.testing.assert_allclose(physical_reward, legacy_reward[0], atol=1e-7)
    np.testing.assert_array_equal(physical_done, legacy_done[0])
    for name in ("step_count", "prev_dist_to_goal", "episode_returns", "episode_lengths"):
        np.testing.assert_allclose(getattr(stationary.crowd, name), getattr(legacy, name), atol=1e-7)
    result, _, _, _ = step(initial, jnp.asarray(command))
    np.testing.assert_allclose(result.physics.qpos, native.qpos, atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(result.physics.qvel, native.qvel, atol=2e-3, rtol=2e-3)
    for _ in range(19):
        result, _, _, metrics = step(result, jnp.array([1., 0.], jnp.float32))
    assert metrics["tracking_speed_error"] < .06
    for _ in range(10):
        result, _, _, metrics = step(result, jnp.zeros(2, jnp.float32))
    assert np.linalg.norm(result.physics.qvel[:2]) < .01

    scenes = []
    for distance in (.65, .4):
        positions = crowd.human_positions.at[0, 0].set(jnp.array([distance, 0.], jnp.float32))
        scenes.append(crowd._replace(human_mask=crowd.human_mask.at[0, 0].set(1.),
            human_positions=positions, human_goals=positions, human_vels=jnp.zeros_like(crowd.human_vels)))
    scenes += [crowd._replace(goal_pos=jnp.array([[.3, 0.]], jnp.float32)),
               crowd._replace(step_count=jnp.array([59], jnp.int32))]
    batch = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *scenes)
    states = jax.jit(jax.vmap(env.from_crowd))(batch, jax.random.split(key, 4))
    next_states, rewards, done, m = jax.jit(jax.vmap(env.step))(states, jnp.zeros((4, 2), jnp.float32))
    np.testing.assert_array_equal(m["collision"][:2], [1., 1.])
    np.testing.assert_array_equal(m["physical_contact"][:2], [0., 1.])
    assert m["success"][2] == 1 and m["terminated"][2] == 1
    assert m["timeout"][3] == 1 and m["terminated"][3] == 0
    np.testing.assert_array_equal(done, np.ones(4))
    assert np.isfinite(rewards).all()
    assert np.all(env.model.geom_type[env.crowd_geom_ids] == mujoco.mjtGeom.mjGEOM_CYLINDER)


def test_compiled_autoreset_and_wandb_payload(tmp_path, monkeypatch):
    import wandb
    from scripts.run_ppo_base import build_trainer
    config = make_config(robot="point", extra=["env=crax_point_crowd", "env.max_steps=1",
        f"save_dir={tmp_path}", "run_name=crax_reset_test", "trainer.num_envs=2",
        "trainer.timesteps_per_batch=4", "trainer.total_timesteps=8", "trainer.num_minibatches=2",
        "trainer.n_updates_per_iteration=1", "trainer.finite_check_interval=1",
        "trainer.eval_interval=1", "trainer.eval_episodes=1", "wandb_interval=1"])
    trainer = build_trainer(config)
    calls = []
    monkeypatch.setattr(wandb, "log", lambda payload, step: calls.append((payload, step)))
    trainer.train()
    state, history, running_returns = trainer.train_state.env_state
    np.testing.assert_array_equal(state.crowd.step_count, np.zeros((2, 1)))
    np.testing.assert_array_equal(running_returns, np.zeros(2))
    np.testing.assert_array_equal(history[:, 0], history[:, -1])
    np.testing.assert_array_equal(history[:, 0], trainer.crax_env.batch_observe(None, state))
    assert not np.array_equal(state.rng[0], state.rng[1])
    assert len(calls) == 2
    for payload, step in calls:
        assert step in (4, 8)
        for key in ("control/tracking_speed_error", "control/tracking_omega_error",
            "control/actuator_saturation", "control/physical_contact", "diagnostics/alpha_active_min",
            "diagnostics/qp_slack_mean", "charts/eval_timeout_rate", "train/timeout_rate"):
            assert key in payload and np.isfinite(payload[key])
        assert payload["train/timeout_rate"] == 1.

    from scripts.eval import Evaluator
    import imageio_ffmpeg
    import json
    import subprocess
    checkpoint = sorted(trainer.save_dir.glob("ckpt_*.pkl"))[-1]
    plain = Evaluator(trainer.save_dir, seeds=[100], episodes=1).eval_all_ckpts(checkpoint)
    rendered = Evaluator(trainer.save_dir, visualize=True, seeds=[100], episodes=1).eval_all_ckpts(checkpoint)
    for name, value in plain[checkpoint.name].items():
        if value is None:
            assert rendered[checkpoint.name][name] is None
        else:
            np.testing.assert_allclose(rendered[checkpoint.name][name], value, rtol=1e-5, atol=1e-6)
    video = next(trainer.save_dir.glob("visualize_*/*.mp4"))
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(video),
        "-vf", "scale=32:32", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, check=True)
    assert len(decoded.stdout) >= 2 * 32 * 32 * 3
    records = json.loads((video.parent / "episodes.json").read_text())
    assert records[0]["seed"] == 100
    assert records[0]["episode_length"] == rendered[checkpoint.name]["mean_episode_length"]
