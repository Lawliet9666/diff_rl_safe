"""Functional CRAX/MJX Point with the original crowd-navigation task.

Only scene construction uses Python. Reset, crowd motion, feedback control,
physics, rewards and observations are pure JAX functions, suitable for vmap/scan.
"""
from __future__ import annotations

import sys
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from crax.envs.safe_goal import SafeGoalPoint
from crax.mjx import collisions
from mujoco.mjx._src.collision_driver import _COLLISION_FUNC

from env.obstacle.obstacle import advance_obstacles
from env.jax_env.jax_social_nav import (
    compute_distances, config_from_omegaconf, fixed_order_policy_obs_from_state,
    reset_env, reward_done,
)


def make_crowd_kernels(cfg):
    """Reuse source obstacle motion, reward arithmetic and observation signs."""
    def advance(state, rng):
        return advance_obstacles(cfg, state, rng)

    def finish(state, command, clearance):
        distance, _ = compute_distances(cfg, state)
        reward, done, success, collision, timeout, step, boundary = reward_done(
            cfg, state, distance, clearance, command, jnp.zeros((1,), jnp.float32))
        total = state.episode_returns + reward
        length = state.episode_lengths + 1
        state = state._replace(step_count=step, prev_dist_to_goal=distance,
                              episode_returns=total, episode_lengths=length)
        return state, reward, success, collision, timeout, boundary

    return {
        "advance": jax.jit(advance), "finish": jax.jit(finish),
        "observe": jax.jit(lambda state: fixed_order_policy_obs_from_state(cfg, state)),
        "reset": jax.jit(lambda rng: reset_env(cfg, rng, 1)),
    }


class CrowdState(NamedTuple):
    physics: Any
    crowd: Any
    rng: Any


# Keep MJX/CRAX array creation float32 while policy/QP computations may use x64.
class _PhysicsNumpy:
    _constructors = {"array", "asarray", "zeros", "ones", "full", "eye",
                     "empty", "zeros_like", "ones_like", "full_like", "empty_like",
                     "identity", "linspace", "logspace", "geomspace", "arange"}

    def __getattr__(self, name):
        fn = getattr(jnp, name)
        if name not in self._constructors:
            return fn

        def create(*args, **kwargs):
            result = fn(*args, **kwargs)
            if hasattr(result, "dtype") and result.dtype == jnp.float64:
                return result.astype(jnp.float32)
            return result
        return create


def configure_mjx_float32():
    proxy = _PhysicsNumpy()
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("mujoco.mjx.", "crax.mjx.")) or name in {"crax.base", "crax.math"}:
            for alias in ("jp", "jnp"):
                if getattr(module, alias, None) is jnp:
                    setattr(module, alias, proxy)


def _float32(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float32) if hasattr(x, "dtype")
        and np.issubdtype(x.dtype, np.floating) else x, tree)


class CraxSocialNav(SafeGoalPoint):
    """CRAX Point scene and MJX pipeline; our task, controller and observations."""

    def __init__(self, config):
        if config.robot.get("simulator_model") != "Point" or config.robot.type != "unicycle":
            raise ValueError("crax_social_nav requires robot=point")
        configure_mjx_float32()
        self.nav_cfg = cfg = config_from_omegaconf(config)
        if cfg.omega_delta_factor:
            raise ValueError("Point requires omega_delta_factor=0")
        self.physics_dt = float(config.env.simulation.physics_dt)
        self.substeps = round(cfg.dt / self.physics_dt)
        if self.substeps <= 0 or not np.isclose(self.substeps * self.physics_dt, cfg.dt):
            raise ValueError("env.dt must be an integer multiple of physics_dt")
        self.tracking_tau = float(config.env.simulation.tracking_time_constant)
        if self.tracking_tau <= 0:
            raise ValueError("tracking_time_constant must be positive")
        self.render_size = int(config.env.simulation.render_size)
        # Importing the explicit extension must succeed; never fall back to boxes.
        box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
        if _COLLISION_FUNC.get((box, cyl)) is not collisions.box_cylinder:
            raise RuntimeError("CRAX box/cylinder collision extension was not registered")
        super().__init__(backend="mjx", timestep=self.physics_dt, n_frames=1,
            episode_length=cfg.max_steps, reset_noise_scale=0.,
            placement_extents=(*np.asarray(cfg.workspace_low), *np.asarray(cfg.workspace_high)),
            hazard_specs=[dict(type="cylinder", count=cfg.num_humans, size=cfg.human_radius,
                height=.2, collidable=True, fixed=False, density=1.,
                positions=[(30. + i * 2., 30., .2) for i in range(cfg.num_humans)])],
            goal_type="cylinder", goal_size=cfg.goal_tolerance, goal_height=.01,
            goal_positions=[(0., 0., .01)])
        self.sys = _float32(self.sys)
        self.model = self.sys.mj_model
        self.crowd_geom_ids = np.array([self.model.geom(f"hazard{i+1}").id for i in range(cfg.num_humans)])
        if not np.all(self.model.geom_type[self.crowd_geom_ids] == cyl):
            raise RuntimeError("Crowd geometry must remain cylindrical")
        self.crowd_mocap_ids = np.array([self.model.body(f"hazard{i+1}").mocapid[0] for i in range(cfg.num_humans)])
        self.goal_mocap_id = int(self.model.body("goal1").mocapid[0])
        self.agent_geom_ids = np.flatnonzero(self.model.geom_bodyid == self.model.body("agent").id)
        self.kernels = make_crowd_kernels(cfg)
        self._template = _float32(jax.jit(lambda: self.pipeline_init(
            jnp.asarray(self.model.qpos0, jnp.float32), jnp.zeros(self.model.nv, jnp.float32)))())
        self._gears = jnp.asarray(self.model.actuator_gear[:2, 0], jnp.float32)
        self._force_range = jnp.asarray(self.model.actuator_forcerange[:2], jnp.float32)
        self._ctrl_range = jnp.asarray(self.model.actuator_ctrlrange[:2], jnp.float32)
        self._gain = jnp.float32(self.model.actuator_gainprm[1, 0])
        self._bounds = jnp.asarray([cfg.robot_vmax, cfg.robot_omega_max], jnp.float32)
        self._reset_with_keys = jax.jit(jax.vmap(self.reset), inline=True)

    def _mocap(self, physics, crowd, positions):
        active = crowd.human_mask[0] > .5
        # Moving masked geoms outside the scene preserves fixed model topology.
        hidden = jnp.stack([1000. + jnp.arange(self.nav_cfg.num_humans) * 2.,
                            jnp.full((self.nav_cfg.num_humans,), 1000.)], axis=-1)
        xy = jnp.where(active[:, None], positions, hidden).astype(jnp.float32)
        pos = physics.mocap_pos.at[self.crowd_mocap_ids, :2].set(xy)
        pos = pos.at[self.crowd_mocap_ids, 2].set(jnp.float32(.2))
        pos = pos.at[self.goal_mocap_id, :2].set(crowd.goal_pos[0])
        return physics.replace(mocap_pos=pos)

    def from_crowd(self, crowd, key):
        """Initialize physics from an existing task state for parity/scenario checks."""
        physics = self._template.replace(qpos=crowd.robot_state[0].astype(jnp.float32),
            qvel=jnp.concatenate([crowd.robot_vel[0], crowd.robot_action[0, 1:2]]).astype(jnp.float32))
        physics = self._mocap(physics, crowd, crowd.human_positions[0])
        physics = mjx.forward(self.sys, physics)
        return CrowdState(physics, crowd, key)

    def reset(self, key):
        key, crowd_key = jax.random.split(key)
        return self.from_crowd(self.kernels["reset"](crowd_key), key)

    def observe(self, state):
        return self.kernels["observe"](state.crowd)[0]

    def tracking_control(self, physics, command):
        """Same inverse-mass speed/turn tracker as the native Point backend."""
        yaw = physics.qpos[2]
        heading = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)])
        speed, omega = physics.qvel[:2] @ heading, physics.qvel[2]
        acceleration = jnp.concatenate([heading * (command[0] - speed) / self.tracking_tau,
                                        ((command[1] - omega) / self.tracking_tau)[None]])
        mass = mjx.full_m(self.sys, physics)
        force = mass @ acceleration + physics.qfrc_bias - physics.qfrc_passive
        desired = jnp.stack([force[:2] @ heading / self._gears[0], force[2] / self._gears[1]])
        limited = jnp.clip(desired, self._force_range[:, 0], self._force_range[:, 1])
        ctrl = jnp.stack([limited[0], self._gears[1] * omega + limited[1] / self._gain])
        clipped = jnp.clip(ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        return clipped

    def _contact(self, physics):
        contact = physics._impl.contact
        a, b = contact.geom1, contact.geom2
        agent_a = jnp.any(a[:, None] == self.agent_geom_ids, axis=1)
        agent_b = jnp.any(b[:, None] == self.agent_geom_ids, axis=1)
        crowd_a = jnp.any(a[:, None] == self.crowd_geom_ids, axis=1)
        crowd_b = jnp.any(b[:, None] == self.crowd_geom_ids, axis=1)
        return jnp.any(((agent_a & crowd_b) | (agent_b & crowd_a)) & (contact.dist <= 0.))

    def _clearance(self, physics, crowd):
        distances = (jnp.linalg.norm(crowd.human_positions[0] - physics.qpos[:2], axis=-1)
                     - crowd.human_radii[0] - self.nav_cfg.robot_radius)
        return jnp.min(jnp.where(crowd.human_mask[0] > .5, distances, jnp.inf))

    def step(self, state, velocity_command):
        command = jnp.clip(jnp.asarray(velocity_command, jnp.float32), -self._bounds, self._bounds)
        key, motion_key = jax.random.split(state.rng)
        crowd = state.crowd
        # Obstacles stay at their previous positions throughout robot physics.
        physics = self._mocap(state.physics, crowd, crowd.human_positions[0])

        def substep(carry, _):
            physics, clearance, contacted = carry
            # reset/final forward and the preceding substep already refresh physics.
            ctrl = self.tracking_control(physics, command)
            physics = self.pipeline_step(physics, ctrl)
            physics = mjx.forward(self.sys, physics)
            minimum = self._clearance(physics, crowd)
            return (physics, jnp.minimum(clearance, minimum), contacted | self._contact(physics)), None

        (physics, clearance, physical), _ = jax.lax.scan(substep,
            (physics, jnp.float32(jnp.inf), jnp.bool_(False)),
            None, length=self.substeps)
        yaw = jnp.arctan2(jnp.sin(physics.qpos[2]), jnp.cos(physics.qpos[2]))
        velocity = physics.qvel[:2]
        heading = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)])
        actual = jnp.stack([velocity @ heading, physics.qvel[2]])
        crowd = crowd._replace(robot_state=jnp.concatenate([physics.qpos[:2], yaw[None]])[None],
                               robot_vel=velocity[None], robot_action=actual[None])
        crowd = self.kernels["advance"](crowd, motion_key)
        physics = self._mocap(physics, crowd, crowd.human_positions[0])
        physics = mjx.forward(self.sys, physics)
        clearance = jnp.minimum(clearance, self._clearance(physics, crowd))
        physical = physical | self._contact(physics)
        crowd, reward, success, collision, timeout, boundary = self.kernels["finish"](
            crowd, command[None], clearance[None])
        terminated = jnp.maximum(success[0], collision[0])
        done = jnp.maximum(terminated, timeout[0])
        metrics = dict(success=success[0], collision=collision[0], timeout=timeout[0],
            terminated=terminated, truncated=timeout[0], min_clearance=clearance,
            physical_contact=physical.astype(jnp.float32),
            boundary_violation=boundary[0], episode_return=crowd.episode_returns[0],
            episode_length=crowd.episode_lengths[0], robot_omega_delta=jnp.float32(0.),
            robot_omega_delta_penalty=jnp.float32(0.))
        return CrowdState(physics, crowd, key), reward[0], done, metrics

    def reset_fn(self, env_cfg, key, num_envs):
        """Reset a batch from a root key or explicit per-episode keys."""
        if key.shape == (2,):
            key = jax.random.split(key, num_envs)
        elif key.shape != (num_envs, 2):
            raise ValueError("reset keys must have shape (2,) or (batch_size, 2)")
        return self._reset_with_keys(key)

    def step_fn(self, env_cfg, state, action, key):
        return jax.vmap(self.step)(state, action)

    def observe_fn(self, env_cfg, state, hparams=None):
        return jax.vmap(self.observe)(state)

    @staticmethod
    def snapshot_fn(state):
        return dict(qpos=state.physics.qpos, qvel=state.physics.qvel,
                    mocap_pos=state.physics.mocap_pos, mask=state.crowd.human_mask[:, 0])

    def save_videos(self, video_dir, traces, records, env_cfg, hparams, module):
        from env.crax_env.crax_render import save_videos
        return save_videos(self, video_dir, traces, records, env_cfg, hparams, module)
