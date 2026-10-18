"""Deterministic moving encounters and same-model native/MJX contact onset."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from trainer.jax_platform import configure_jax_platform_for_device
    configure_jax_platform_for_device("cuda")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from env.crax_point_crowd import CraxPointCrowd
    from env.jax_social_nav import reset_env
    OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
    OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/"config")):
        config = compose(config_name="config", overrides=["env=crax_point_crowd", "robot=point", "device=cuda"])
    env = CraxPointCrowd(config)
    key = jax.random.PRNGKey(7)
    base = reset_env(env.nav_cfg, key, 1)
    base = base._replace(robot_state=jnp.zeros((1, 3), jnp.float32),
        robot_vel=jnp.zeros((1, 2), jnp.float32), robot_action=jnp.zeros((1, 2), jnp.float32),
        goal_pos=jnp.array([[5., 0.]], jnp.float32), prev_dist_to_goal=jnp.array([5.], jnp.float32),
        human_mask=jnp.zeros_like(base.human_mask).at[0, 0].set(1.),
        human_vmaxs=jnp.zeros_like(base.human_vmaxs), human_vels=jnp.zeros_like(base.human_vels))
    scenes = []
    for robot, obstacle, goal, speed in [([0., 0.], [3., 0.], [-10., 0.], 1.),
        ([-2., 0.], [0., -2.], [0., 10.], 1.), ([0., 0.], [2., 0.], [10., 0.], .5)]:
        scene = base._replace(robot_state=base.robot_state.at[0, :2].set(jnp.array(robot, jnp.float32)),
            human_positions=base.human_positions.at[0, 0].set(jnp.array(obstacle, jnp.float32)),
            human_goals=base.human_goals.at[0, 0].set(jnp.array(goal, jnp.float32)),
            human_vmaxs=base.human_vmaxs.at[0, 0].set(speed))
        scenes.append(scene)
    batch = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *scenes)
    states = jax.jit(jax.vmap(env.from_crowd))(batch, jax.random.split(key, 3))
    step = jax.jit(jax.vmap(env.step))
    commands = jnp.array([[1., 0.], [1., 0.], [1.5, 0.]], jnp.float32)
    collision_steps = np.full(3, -1)
    for i in range(40):
        states, _, _, metrics = step(states, commands)
        collided = np.asarray(metrics["collision"]) > .5
        collision_steps[(collision_steps < 0) & collided] = i+1
    assert np.all(collision_steps > 0), collision_steps
    report = {name: {"collision_step": int(value), "constant_command": list(map(float, commands[i]))}
              for i, (name, value) in enumerate(zip(["head_on", "crossing", "overtaking"], collision_steps))}

    # Sweep actual geometry contact before integration, without a planning envelope.
    distances = np.linspace(.7, .35, 71).astype(np.float32)
    scenes = [base._replace(human_positions=base.human_positions.at[0, 0].set(jnp.array([x, 0.], jnp.float32)))
              for x in distances]
    batch = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *scenes)
    physics = jax.jit(jax.vmap(env.from_crowd))(batch, jax.random.split(key, len(distances)))
    mjx_touch = np.asarray(jax.jit(jax.vmap(lambda s: env._contact(s.physics)))(physics))
    native_touch = []
    native = mujoco.MjData(env.model)
    for index in range(len(distances)):
        native.qpos[:] = physics.physics.qpos[index]
        native.mocap_pos[:] = physics.physics.mocap_pos[index]
        mujoco.mj_forward(env.model, native)
        touched = False
        for c in native.contact[:native.ncon]:
            touched |= c.dist <= 0. and ((c.geom1 in env.agent_geom_ids and c.geom2 in env.crowd_geom_ids)
                or (c.geom2 in env.agent_geom_ids and c.geom1 in env.crowd_geom_ids))
        native_touch.append(touched)
    native_onset, mjx_onset = float(distances[np.argmax(native_touch)]), float(distances[np.argmax(mjx_touch)])
    assert any(native_touch) and mjx_touch.any()
    assert abs(native_onset-mjx_onset) <= .02
    report["contact_onset"] = dict(native=native_onset, mjx=mjx_onset,
        sweep_resolution_m=.005, tolerance_m=.02, engine_version=mujoco.__version__)
    path = Path("outputs/crax_validation/encounters.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
