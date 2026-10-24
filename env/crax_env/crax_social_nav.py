"""CRAX/MJX social navigation with JAX reset, step and observation functions."""
from __future__ import annotations

import sys
import math
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from crax.envs.safe_goal import SafeGoalPoint
from crax.mjx import collisions
from mujoco.mjx._src.collision_driver import _COLLISION_FUNC

from env.robots import Robot, angle_normalize, make_robot


class CrowdState(NamedTuple):
    physics: Any
    crowd: Any
    rng: Any


# MJX uses float32; policy/QP may use float64.
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

class CraxSocialNavConfig(NamedTuple):
    dt: float
    max_steps: int
    workspace_low: jnp.ndarray
    workspace_high: jnp.ndarray
    robot_type: str
    goal_tolerance: float
    robot_radius: float
    robot: Robot
    robot_goal_min_distance: float
    robot_goal_max_tries: int
    num_humans: int
    min_num_humans: int
    random_num_humans: bool
    human_vmax_min: float
    human_vmax_max: float
    human_radius: float
    human_init_noise_range: float
    human_max_tries: int
    human_use_gmm: bool
    gmm_weights: jnp.ndarray
    gmm_stds: jnp.ndarray
    gmm_lateral_ratio: float
    random_goal_changing: bool
    goal_change_chance: float
    end_goal_changing: bool
    end_goal_change_chance: float
    social_force_ki: float
    social_force_a: float
    social_force_b: float
    social_force_avoid_robot: bool
    social_force_avoid_robot_ratio: float
    success_reward: float
    collision_penalty: float
    discomfort_dist: float
    discomfort_penalty_factor: float
    potential_factor: float
    progress_speed_scale: float
    back_factor: float
    spin_factor: float
    omega_delta_factor: float
    constant_penalty: float
    boundary_penalty_factor: float
    boundary_buffer: float


class CraxNavigationState(NamedTuple):
    """Navigation state paired with MJX physics."""
    robot_state: jnp.ndarray
    robot_vel: jnp.ndarray
    robot_action: jnp.ndarray
    goal_pos: jnp.ndarray
    human_positions: jnp.ndarray
    human_vels: jnp.ndarray
    human_goals: jnp.ndarray
    human_vmaxs: jnp.ndarray
    human_radii: jnp.ndarray
    human_mask: jnp.ndarray
    human_avoid_robot_mask: jnp.ndarray
    step_count: jnp.ndarray
    prev_dist_to_goal: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray


class CraxSocialNav(SafeGoalPoint):
    """CRAX physics with the crowd-navigation task."""

    @classmethod
    def config_from_omegaconf(cls, config, *, robot=None) -> CraxSocialNavConfig:
        def get(cfg, name, default=None):
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            get_fn = getattr(cfg, "get", None)
            if callable(get_fn):
                return get_fn(name, default)
            return getattr(cfg, name, default)

        env_cfg = config.env
        robot_cfg = config.robot
        scenario = str(get(env_cfg, "scenario"))
        supported_scenarios = {"crax_social_nav"}
        if scenario not in supported_scenarios:
            raise ValueError(
                f"unsupported env.scenario={scenario!r}; expected one of {tuple(sorted(supported_scenarios))}"
            )
        human_cfg = env_cfg.humans
        if str(get(human_cfg, "human_policy", "social_force")) != "social_force":
            raise ValueError("CRAX environment requires env.humans.human_policy=social_force")
        reward = env_cfg.reward
        social_force = env_cfg.social_force
        gmm = human_cfg.gmm
        workspace = env_cfg.workspace
        workspace_low = np.asarray([float(workspace.x[0]), float(workspace.y[0])], dtype=np.float32)
        workspace_high = np.asarray([float(workspace.x[1]), float(workspace.y[1])], dtype=np.float32)
        workspace_size = workspace_high - workspace_low
        if np.any(workspace_size <= 0.0):
            raise ValueError("env.workspace high bounds must be greater than low bounds")
        vmax = list(human_cfg.vmax)
        num_humans = int(human_cfg.num_humans)
        min_num_humans = int(get(human_cfg, "min_num_humans", num_humans))
        if num_humans <= 0:
            raise ValueError("env.humans.num_humans must be > 0")
        if min_num_humans < 0 or min_num_humans > num_humans:
            raise ValueError("env.humans.min_num_humans must satisfy 0 <= min_num_humans <= num_humans")
        avoid_robot_ratio = float(get(social_force, "avoid_robot_ratio", 1.0))
        if avoid_robot_ratio < 0.0 or avoid_robot_ratio > 1.0:
            raise ValueError("env.social_force.avoid_robot_ratio must satisfy 0 <= ratio <= 1")
        robot_type = str(robot_cfg.type).strip().lower()
        if robot is None:
            robot = make_robot(config)
        progress_speed_scale = float(reward.progress_speed_scale)
        if not np.isfinite(progress_speed_scale) or progress_speed_scale <= 0:
            raise ValueError("env.reward.progress_speed_scale must be finite and positive")
        omega_delta_factor = float(get(reward, "omega_delta_factor", 0.0))
        if omega_delta_factor < 0.0:
            raise ValueError("env.reward.omega_delta_factor must be >= 0")
        if omega_delta_factor > 0.0 and robot_type != "unicycle":
            raise ValueError("env.reward.omega_delta_factor requires robot.type=unicycle")
        dt = float(env_cfg.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("env.dt must be positive")
        return CraxSocialNavConfig(
            dt=dt,
            max_steps=int(env_cfg.max_steps),
            workspace_low=jnp.asarray(workspace_low, dtype=jnp.float32),
            workspace_high=jnp.asarray(workspace_high, dtype=jnp.float32),
            robot_type=robot_type,
            goal_tolerance=float(env_cfg.robot_goal.tolerance),
            robot_radius=float(robot_cfg.radius),
            robot=robot,
            robot_goal_min_distance=float(env_cfg.robot_goal.min_distance),
            robot_goal_max_tries=int(env_cfg.robot_goal.max_tries),
            num_humans=num_humans,
            min_num_humans=min_num_humans,
            random_num_humans=bool(get(human_cfg, "random_num_humans", False)),
            human_vmax_min=float(vmax[0]),
            human_vmax_max=float(vmax[1]),
            human_radius=float(human_cfg.radius),
            human_init_noise_range=float(human_cfg.init_noise_range),
            human_max_tries=int(human_cfg.max_tries),
            human_use_gmm=bool(human_cfg.use_gmm),
            gmm_weights=jnp.asarray(gmm.weights, dtype=jnp.float32),
            gmm_stds=jnp.asarray(gmm.stds, dtype=jnp.float32),
            gmm_lateral_ratio=float(gmm.lateral_ratio),
            random_goal_changing=bool(get(human_cfg, "random_goal_changing", False)),
            goal_change_chance=float(np.clip(get(human_cfg, "goal_change_chance", 0.0), 0.0, 1.0)),
            end_goal_changing=bool(human_cfg.end_goal_changing),
            end_goal_change_chance=float(np.clip(get(human_cfg, "end_goal_change_chance", 1.0), 0.0, 1.0)),
            social_force_ki=float(social_force.KI),
            social_force_a=float(social_force.A),
            social_force_b=float(social_force.B),
            social_force_avoid_robot=bool(social_force.avoid_robot),
            social_force_avoid_robot_ratio=avoid_robot_ratio,
            success_reward=float(reward.success_reward),
            collision_penalty=float(reward.collision_penalty),
            discomfort_dist=float(reward.discomfort_dist),
            discomfort_penalty_factor=float(reward.discomfort_penalty_factor),
            potential_factor=float(reward.potential_factor),
            progress_speed_scale=progress_speed_scale,
            back_factor=float(get(reward, "back_factor", 0.0)),
            spin_factor=float(get(reward, "spin_factor", 0.0)),
            omega_delta_factor=omega_delta_factor,
            constant_penalty=float(reward.constant_penalty),
            boundary_penalty_factor=float(reward.boundary_penalty_factor),
            boundary_buffer=float(reward.boundary_buffer),
        )

    def __init__(self, config):
        if config.robot.type != "unicycle":
            raise ValueError("crax_social_nav requires robot.type=unicycle")
        configure_mjx_float32()
        self.nav_cfg = cfg = self.config_from_omegaconf(config)
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
        self.kernels = {
            "advance": jax.jit(lambda state, rng: self._advance_obstacles(cfg, state, rng)),
            "finish": jax.jit(self._finish_crowd),
            "observe": jax.jit(lambda state: self._fixed_order_policy_obs_from_state(cfg, state)),
            "reset": jax.jit(self._reset_crowd),
        }
        self._template = _float32(jax.jit(lambda: self.pipeline_init(
            jnp.asarray(self.model.qpos0, jnp.float32), jnp.zeros(self.model.nv, jnp.float32)))())
        self._gears = jnp.asarray(self.model.actuator_gear[:2, 0], jnp.float32)
        self._force_range = jnp.asarray(self.model.actuator_forcerange[:2], jnp.float32)
        self._ctrl_range = jnp.asarray(self.model.actuator_ctrlrange[:2], jnp.float32)
        self._gain = jnp.float32(self.model.actuator_gainprm[1, 0])
        self._bounds = jnp.asarray(cfg.robot.input_bounds()[1], jnp.float32)
        self._reset_with_keys = jax.jit(jax.vmap(self.reset), inline=True)

    def _mocap(self, physics, crowd, positions):
        active = crowd.human_mask[0] > .5
        # Keep fixed topology by moving masked geoms outside the scene.
        hidden = jnp.stack([1000. + jnp.arange(self.nav_cfg.num_humans) * 2.,
                            jnp.full((self.nav_cfg.num_humans,), 1000.)], axis=-1)
        xy = jnp.where(active[:, None], positions, hidden).astype(jnp.float32)
        pos = physics.mocap_pos.at[self.crowd_mocap_ids, :2].set(xy)
        pos = pos.at[self.crowd_mocap_ids, 2].set(jnp.float32(.2))
        pos = pos.at[self.goal_mocap_id, :2].set(crowd.goal_pos[0])
        return physics.replace(mocap_pos=pos)

    def from_crowd(self, crowd, key):
        """Initialize physics from an existing navigation state."""
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
        """Native Point speed/yaw-rate tracking control."""
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
        # Hold obstacles fixed during robot physics.
        physics = self._mocap(state.physics, crowd, crowd.human_positions[0])

        def substep(carry, _):
            physics, clearance, contacted = carry
            # Physics was refreshed by reset or the previous substep.
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
        """Reset from a root key or per-episode keys."""
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

    def compute_distances(self, cfg: CraxSocialNavConfig, state: CraxNavigationState):
        dist_to_goal = jnp.linalg.norm(state.robot_state[:, 0:2] - state.goal_pos, axis=-1)
        clearances = (
            jnp.linalg.norm(state.human_positions - state.robot_state[:, None, 0:2], axis=-1)
            - cfg.robot_radius
            - state.human_radii
        )
        clearances = jnp.where(state.human_mask > 0.5, clearances, jnp.inf)
        return dist_to_goal, jnp.min(clearances, axis=-1)

    def robot_boundary_violation(self, cfg: CraxSocialNavConfig, state: CraxNavigationState):
        pos = state.robot_state[:, 0:2]
        low_violation = cfg.workspace_low[None, :] - cfg.boundary_buffer - pos
        high_violation = pos - (cfg.workspace_high[None, :] + cfg.boundary_buffer)
        violations = jnp.concatenate([low_violation, high_violation], axis=1)
        return jnp.maximum(jnp.max(violations, axis=1), 0.0).astype(jnp.float32)

    def navigation_done(self, cfg: CraxSocialNavConfig, state: CraxNavigationState,
                        dist_to_goal, min_clearance):
        next_step = state.step_count + 1
        collision = min_clearance < 0.0
        success = dist_to_goal < cfg.goal_tolerance
        timeout = (next_step >= cfg.max_steps) & ~(collision | success)
        done = collision | success | timeout
        return (done, success, collision, timeout, next_step,
                self.robot_boundary_violation(cfg, state))

    def reward_done(self,
        cfg: CraxSocialNavConfig,
        state: CraxNavigationState,
        dist_to_goal,
        min_clearance,
        action,
        omega_delta_penalty,
    ):
        done, success, collision, timeout, next_step, boundary_violation = self.navigation_done(
            cfg, state, dist_to_goal, min_clearance)
        discomfort = (min_clearance < cfg.discomfort_dist) & ~(collision | success | timeout)
        potential = ~(collision | success | timeout | discomfort)

        potential_reward = cfg.potential_factor * (state.prev_dist_to_goal - dist_to_goal)
        potential_reward = jnp.clip(
            potential_reward,
            -cfg.dt * cfg.progress_speed_scale * cfg.potential_factor,
            cfg.dt * cfg.progress_speed_scale * cfg.potential_factor,
        )
        reward = jnp.where(collision, cfg.collision_penalty, 0.0)
        reward = jnp.where(success, cfg.success_reward, reward)
        reward = jnp.where(timeout, 0.0, reward)
        reward = jnp.where(discomfort, (min_clearance - cfg.discomfort_dist) * cfg.discomfort_penalty_factor * cfg.dt, reward)
        reward = jnp.where(potential, potential_reward, reward)

        spin = -cfg.spin_factor * jnp.square(action[:, 1] * cfg.dt)
        back = jnp.where(action[:, 0] < 0.0, -cfg.back_factor * jnp.abs(action[:, 0]), 0.0)
        boundary_penalty = cfg.boundary_penalty_factor * boundary_violation
        reward = (
            reward
            + spin
            + back
            + cfg.constant_penalty
            - boundary_penalty
            - omega_delta_penalty
        ) / 10.0
        return reward, done, success, collision, timeout, next_step, boundary_violation

    def _workspace_rectangle_samples(self, cfg: CraxSocialNavConfig, rng, shape):
        unit = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
        return cfg.workspace_low + unit * (cfg.workspace_high - cfg.workspace_low)

    def _planar_state(self, cfg, state):
        """Project the robot state to [x, y, yaw] for navigation and rendering."""
        return state._replace(robot_state=jax.vmap(cfg.robot.planar_pose)(state.robot_state))

    def _fixed_order_policy_obs_from_state(self, cfg: CraxSocialNavConfig, state: CraxNavigationState):
        state = self._planar_state(cfg, state)
        batch_size = state.robot_state.shape[0]
        robot = jnp.concatenate(
            [
                state.robot_state[:, 0:2] - state.goal_pos,
                state.robot_vel,
                state.robot_state[:, 2:3],
                jnp.full((batch_size, 1), cfg.robot_radius, dtype=state.robot_state.dtype),
            ],
            axis=1,
        )
        blocks = jnp.concatenate(
            [
                state.robot_state[:, None, 0:2] - state.human_positions,
                state.human_vels,
                state.human_radii[..., None],
                state.human_mask[..., None],
            ],
            axis=-1,
        )
        return jnp.concatenate([robot, blocks.reshape((batch_size, -1))], axis=1).astype(jnp.float32)

    def _reset_one(self, cfg: CraxSocialNavConfig, rng):
        key_robot, key_theta, key_goal, key_humans, key_avoid_robot = jax.random.split(rng, 5)
        key_active = jax.random.fold_in(rng, 17)
        angle = jax.random.uniform(key_robot, (), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32)
        robot_pos = self._ellipse_points_from_angles(cfg, angle)
        if cfg.robot_type != "single_integrator":
            robot_theta = jax.random.uniform(
                key_theta,
                (),
                minval=0.0,
                maxval=2.0 * jnp.pi,
                dtype=jnp.float32,
            )
        else:
            robot_theta = jnp.asarray(0.0, dtype=jnp.float32)

        center, radii = self._workspace_center_radii(cfg)
        goal_candidates = self._workspace_rectangle_samples(cfg, key_goal, (cfg.robot_goal_max_tries, 2))
        goal_valid = jnp.linalg.norm(goal_candidates - robot_pos[None, :], axis=1) >= cfg.robot_goal_min_distance
        goal_idx = jnp.argmax(goal_valid)
        fallback_goal = 2.0 * center - robot_pos
        goal_pos = jnp.where(jnp.any(goal_valid), goal_candidates[goal_idx], fallback_goal)

        init_positions = jnp.zeros((cfg.num_humans, 2), dtype=jnp.float32)
        init_goals = jnp.zeros((cfg.num_humans, 2), dtype=jnp.float32)
        init_vmaxs = jnp.zeros((cfg.num_humans,), dtype=jnp.float32)
        human_radii = jnp.full((cfg.num_humans,), cfg.human_radius, dtype=jnp.float32)
        safe_pair_dist = 2.0 * cfg.human_radius + cfg.discomfort_dist
        robot_safe_dist = 0.5 * jnp.linalg.norm(radii)
        human_outer_scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))

        def sample_human(carry, idx):
            positions, goals, vmaxs, key = carry
            key, key_angle, key_noise, key_vmax = jax.random.split(key, 4)
            thetas = jax.random.uniform(
                key_angle,
                (cfg.human_max_tries,),
                minval=0.0,
                maxval=2.0 * jnp.pi,
                dtype=jnp.float32,
            )
            noise = jax.random.uniform(
                key_noise,
                (cfg.human_max_tries, 2),
                minval=-0.5 * cfg.human_init_noise_range,
                maxval=0.5 * cfg.human_init_noise_range,
                dtype=jnp.float32,
            )
            candidates = self._ellipse_points_from_angles(cfg, thetas, human_outer_scale) + noise
            candidate_goals = 2.0 * center - candidates
            previous = jnp.arange(cfg.num_humans) < idx
            dist_pos = jnp.linalg.norm(positions[None, :, :] - candidates[:, None, :], axis=2)
            dist_goals = jnp.linalg.norm(goals[None, :, :] - candidates[:, None, :], axis=2)
            clear_previous = jnp.all(
                jnp.where(
                    previous[None, :],
                    (dist_pos >= safe_pair_dist) & (dist_goals >= safe_pair_dist),
                    True,
                ),
                axis=1,
            )
            clear_robot = jnp.linalg.norm(candidates - robot_pos[None, :], axis=1) >= robot_safe_dist
            clear_goal = jnp.linalg.norm(candidates - goal_pos[None, :], axis=1) >= robot_safe_dist
            valid = clear_previous & clear_robot & clear_goal
            choice = jnp.argmax(valid)
            pos = candidates[choice]
            goal = candidate_goals[choice]
            vmax = jax.random.uniform(
                key_vmax,
                (),
                minval=cfg.human_vmax_min,
                maxval=cfg.human_vmax_max,
                dtype=jnp.float32,
            )
            positions = positions.at[idx].set(pos)
            goals = goals.at[idx].set(goal)
            vmaxs = vmaxs.at[idx].set(vmax)
            return (positions, goals, vmaxs, key), None

        (human_positions, human_goals, human_vmaxs, _key), _ = jax.lax.scan(
            sample_human,
            (init_positions, init_goals, init_vmaxs, key_humans),
            jnp.arange(cfg.num_humans),
        )
        if cfg.random_num_humans:
            active_num_humans = jax.random.randint(
                key_active,
                (),
                minval=cfg.min_num_humans,
                maxval=cfg.num_humans + 1,
                dtype=jnp.int32,
            )
        else:
            active_num_humans = jnp.asarray(cfg.num_humans, dtype=jnp.int32)
        human_mask = (jnp.arange(cfg.num_humans) < active_num_humans).astype(jnp.float32)
        human_positions = jnp.where(human_mask[:, None] > 0.5, human_positions, 0.0)
        human_goals = jnp.where(human_mask[:, None] > 0.5, human_goals, 0.0)
        human_vmaxs = jnp.where(human_mask > 0.5, human_vmaxs, 0.0)
        if cfg.social_force_avoid_robot:
            human_avoid_robot_mask = jax.random.bernoulli(
                key_avoid_robot,
                p=cfg.social_force_avoid_robot_ratio,
                shape=(cfg.num_humans,),
            ).astype(jnp.float32)
            human_avoid_robot_mask = jnp.where(human_mask > 0.5, human_avoid_robot_mask, 0.0)
        else:
            human_avoid_robot_mask = jnp.zeros((cfg.num_humans,), dtype=jnp.float32)
        pose = jnp.asarray([robot_pos[0], robot_pos[1], angle_normalize(robot_theta)], dtype=jnp.float32)
        robot_state = cfg.robot.initial_state(pose)
        robot_action = jnp.zeros((cfg.robot.control_dim,), dtype=robot_state.dtype)
        robot_vel = jnp.zeros((2,), dtype=robot_state.dtype)
        prev_dist = jnp.linalg.norm(robot_pos - goal_pos)
        return CraxNavigationState(
            robot_state=robot_state,
            robot_vel=robot_vel,
            robot_action=robot_action,
            goal_pos=goal_pos,
            human_positions=human_positions,
            human_vels=jnp.zeros_like(human_positions),
            human_goals=human_goals,
            human_vmaxs=human_vmaxs,
            human_radii=human_radii,
            human_mask=human_mask,
            human_avoid_robot_mask=human_avoid_robot_mask,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            prev_dist_to_goal=prev_dist.astype(jnp.float32),
            episode_returns=jnp.asarray(0.0, dtype=jnp.float32),
            episode_lengths=jnp.asarray(0, dtype=jnp.int32),
        )

    def _workspace_center_radii(self, cfg):
        center = 0.5 * (cfg.workspace_low + cfg.workspace_high)
        radii = jnp.maximum(0.5 * (cfg.workspace_high - cfg.workspace_low), 1e-3)
        return center, radii

    def _outer_motion_bounds(self, cfg):
        center, radii = self._workspace_center_radii(cfg)
        half_extents = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32)) * radii
        half_extents = half_extents + cfg.human_init_noise_range
        return center - half_extents, center + half_extents

    def _ellipse_points_from_angles(self, cfg, angles, radial_scale=1.0):
        center, radii = self._workspace_center_radii(cfg)
        unit = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        return center + radii * jnp.asarray(radial_scale, dtype=jnp.float32) * unit

    def _gmm_component_means(self, actions, lateral_ratio):
        speed_sq = jnp.sum(actions * actions, axis=-1, keepdims=True)
        zeros = jnp.zeros_like(actions)
        vx = actions[..., 0]
        vy = actions[..., 1]
        scale = 1.0 / jnp.sqrt(1.0 + lateral_ratio * lateral_ratio)
        means = jnp.stack(
            [
                actions,
                scale * jnp.stack([vx - lateral_ratio * vy, vy + lateral_ratio * vx], axis=-1),
                scale * jnp.stack([vx + lateral_ratio * vy, vy - lateral_ratio * vx], axis=-1),
            ],
            axis=-2,
        )
        return jnp.where(speed_sq[..., None] < 1e-12, zeros[..., None, :], means)

    def _gmm_noise_from_component_noise(self, cfg, nominal_actions, component_idx, normal_noise):
        means = self._gmm_component_means(nominal_actions, cfg.gmm_lateral_ratio)
        one_hot = jax.nn.one_hot(component_idx, cfg.gmm_weights.shape[0], dtype=nominal_actions.dtype)
        mean = jnp.sum(means * one_hot[..., None], axis=-2)
        std = jnp.sum(cfg.gmm_stds * one_hot, axis=-1)
        return mean + normal_noise * std[..., None]

    def _apply_gmm_noise(self, cfg, nominal_actions, state, rng):
        comp_key, noise_key = jax.random.split(rng)
        logits = jnp.log(cfg.gmm_weights)
        component_idx = jax.random.categorical(comp_key, logits, shape=nominal_actions.shape[:-1])
        normal_noise = jax.random.normal(noise_key, nominal_actions.shape, dtype=nominal_actions.dtype)
        noisy = self._gmm_noise_from_component_noise(cfg, nominal_actions, component_idx, normal_noise)
        dist_to_goal = jnp.linalg.norm(state.human_goals - state.human_positions, axis=-1)
        noise_mask = (dist_to_goal >= 2.0 * state.human_radii) & (state.human_mask > 0.5)
        return jnp.where(noise_mask[..., None], noisy, nominal_actions)

    def _sample_new_human_goal(self,
        cfg,
        robot_pos,
        robot_goal,
        human_positions,
        human_goals,
        human_radii,
        human_mask,
        human_vmaxs,
        idx,
        rng,
    ):
        old_goal = human_goals[idx]
        v_pref = human_vmaxs[idx]
        self_radius = human_radii[idx]

        key_angle, key_noise = jax.random.split(rng)
        angles = jax.random.uniform(
            key_angle,
            (cfg.human_max_tries,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float32,
        )
        noise = jax.random.uniform(
            key_noise,
            (cfg.human_max_tries, 2),
            minval=-0.5,
            maxval=0.5,
            dtype=jnp.float32,
        ) * v_pref
        human_outer_scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
        candidates = self._ellipse_points_from_angles(cfg, angles, human_outer_scale) + noise

        min_robot_dist = self_radius + cfg.robot_radius + cfg.discomfort_dist
        clear_robot = jnp.linalg.norm(candidates - robot_pos[None, :], axis=1) >= min_robot_dist
        clear_robot_goal = jnp.linalg.norm(candidates - robot_goal[None, :], axis=1) >= min_robot_dist

        other_mask = (jnp.arange(cfg.num_humans) != idx) & (human_mask > 0.5)
        safe_dists = self_radius + human_radii + cfg.discomfort_dist
        dist_to_positions = jnp.linalg.norm(candidates[:, None, :] - human_positions[None, :, :], axis=2)
        dist_to_goals = jnp.linalg.norm(candidates[:, None, :] - human_goals[None, :, :], axis=2)
        clear_humans = jnp.all(
            jnp.where(
                other_mask[None, :],
                (dist_to_positions >= safe_dists[None, :]) & (dist_to_goals >= safe_dists[None, :]),
                True,
            ),
            axis=1,
        )

        valid = clear_robot & clear_robot_goal & clear_humans
        selected = candidates[jnp.argmax(valid)]
        keep_old = (v_pref <= 1e-8) | (~jnp.any(valid))
        return jnp.where(keep_old, old_goal, selected)

    def _update_obstacle_goals(self, cfg, state, rng):
        if not (cfg.end_goal_changing or cfg.random_goal_changing):
            return state
        if cfg.num_humans <= 0:
            return state

        batch_size = state.robot_state.shape[0]
        key_trigger, key_sample = jax.random.split(rng)
        trigger_draws = jax.random.uniform(key_trigger, state.human_vmaxs.shape, dtype=jnp.float32)
        sample_keys = jax.random.split(key_sample, batch_size * cfg.num_humans).reshape((batch_size, cfg.num_humans, 2))
        human_indices = jnp.arange(cfg.num_humans)

        def update_one(robot_state, robot_goal, positions, goals, vmaxs, radii, mask, triggers, keys):
            robot_pos = robot_state[0:2]

            def update_goal(goals_in, idx):
                active = (vmaxs[idx] > 1e-8) & (mask[idx] > 0.5)
                dist_to_goal = jnp.linalg.norm(positions[idx] - goals_in[idx])
                near_goal = dist_to_goal <= radii[idx] + 0.1
                random_trigger = cfg.random_goal_changing & (triggers[idx] <= cfg.goal_change_chance)
                end_goal_trigger = cfg.end_goal_changing & near_goal & (triggers[idx] <= cfg.end_goal_change_chance)
                should_change = active & (random_trigger | end_goal_trigger)

                def change_goal(current_goals):
                    new_goal = self._sample_new_human_goal(
                        cfg,
                        robot_pos,
                        robot_goal,
                        positions,
                        current_goals,
                        radii,
                        mask,
                        vmaxs,
                        idx,
                        keys[idx],
                    )
                    return current_goals.at[idx].set(new_goal)

                return jax.lax.cond(should_change, change_goal, lambda current_goals: current_goals, goals_in), None

            new_goals, _ = jax.lax.scan(update_goal, goals, human_indices)
            return new_goals

        new_goals = jax.vmap(update_one)(
            state.robot_state,
            state.goal_pos,
            state.human_positions,
            state.human_goals,
            state.human_vmaxs,
            state.human_radii,
            state.human_mask,
            trigger_draws,
            sample_keys,
        )
        return state._replace(human_goals=new_goals)

    def _social_force_actions(self, cfg, state):
        pos = state.human_positions
        vel = state.human_vels
        goals = state.human_goals
        radii = state.human_radii
        vpref = jnp.maximum(state.human_vmaxs, 0.0)
        mask = state.human_mask > 0.5

        goal_delta = goals - pos
        dist_goal = jnp.linalg.norm(goal_delta, axis=-1)
        valid_goal = (dist_goal > 1e-5) & mask
        desired_vel = jnp.where(
            valid_goal[..., None],
            goal_delta / jnp.maximum(dist_goal[..., None], 1e-8) * vpref[..., None],
            0.0,
        )
        curr_delta = cfg.social_force_ki * (desired_vel - vel)

        if cfg.social_force_avoid_robot:
            all_pos = jnp.concatenate([pos, state.robot_state[:, None, 0:2]], axis=1)
            all_radii = jnp.concatenate(
                [radii, jnp.full((pos.shape[0], 1), cfg.robot_radius, dtype=pos.dtype)],
                axis=1,
            )
            all_mask = jnp.concatenate([mask, jnp.ones((pos.shape[0], 1), dtype=bool)], axis=1)
            robot_neighbor_mask = jnp.asarray(state.human_avoid_robot_mask > 0.5, dtype=bool)
        else:
            all_pos = pos
            all_radii = radii
            all_mask = mask
            robot_neighbor_mask = jnp.zeros_like(mask, dtype=bool)

        delta = pos[:, :, None, :] - all_pos[:, None, :, :]
        dist = jnp.linalg.norm(delta, axis=-1)
        valid_neighbor = mask[:, :, None] & all_mask[:, None, :]
        eye = jnp.eye(cfg.num_humans, dtype=bool)[None, :, :]
        valid_neighbor = valid_neighbor.at[:, :, : cfg.num_humans].set(valid_neighbor[:, :, : cfg.num_humans] & ~eye)
        if cfg.social_force_avoid_robot:
            valid_neighbor = valid_neighbor.at[:, :, cfg.num_humans].set(
                valid_neighbor[:, :, cfg.num_humans] & robot_neighbor_mask
            )

        b = max(float(cfg.social_force_b), 1e-5)
        sum_r = radii[:, :, None] + all_radii[:, None, :]
        non_overlap = valid_neighbor & (dist > 1e-5)
        force_mag = cfg.social_force_a * jnp.exp((sum_r - dist) / b)
        unit = jnp.where(non_overlap[..., None], delta / jnp.maximum(dist[..., None], 1e-8), 0.0)
        interaction = jnp.sum(force_mag[..., None] * unit * non_overlap[..., None], axis=2)

        new_vel = vel + (curr_delta + interaction) * cfg.dt
        speed = jnp.linalg.norm(new_vel, axis=-1)
        clip_mask = (speed > vpref) & (vpref > 0.0)
        new_vel = jnp.where(clip_mask[..., None], new_vel / jnp.maximum(speed[..., None], 1e-8) * vpref[..., None], new_vel)
        return jnp.where(mask[..., None], new_vel, 0.0)

    def _human_step(self, cfg, state, actions):
        speed = jnp.linalg.norm(actions, axis=-1)
        clipped = jnp.where(
            (state.human_vmaxs > 0.0)[..., None] & (speed > state.human_vmaxs)[..., None],
            actions / jnp.maximum(speed[..., None], 1e-8) * state.human_vmaxs[..., None],
            actions,
        )
        clipped = jnp.where((state.human_vmaxs > 0.0)[..., None], clipped, 0.0)
        clipped = jnp.where((state.human_mask > 0.5)[..., None], clipped, 0.0)
        next_positions = state.human_positions + clipped * cfg.dt
        outer_low, outer_high = self._outer_motion_bounds(cfg)
        clipped_positions = jnp.clip(
            next_positions,
            outer_low[None, None, :],
            outer_high[None, None, :],
        )
        next_positions = jnp.where((state.human_mask > 0.5)[..., None], clipped_positions, state.human_positions)
        return next_positions, clipped

    def _advance_obstacles(self, cfg, state, rng):
        """Update obstacles after robot motion."""
        gmm_rng = rng
        if cfg.end_goal_changing or cfg.random_goal_changing:
            goal_rng, gmm_rng = jax.random.split(rng)
            state = self._update_obstacle_goals(cfg, state, goal_rng)

        human_actions = self._social_force_actions(cfg, state)
        if cfg.human_use_gmm:
            human_actions = self._apply_gmm_noise(cfg, human_actions, state, gmm_rng)
        next_positions, next_vels = self._human_step(cfg, state, human_actions)
        return state._replace(human_positions=next_positions, human_vels=next_vels)

    def _finish_crowd(self, state, command, clearance):
        cfg = self.nav_cfg
        distance, _ = self.compute_distances(cfg, state)
        reward, done, success, collision, timeout, step, boundary = self.reward_done(
            cfg, state, distance, clearance, command, jnp.zeros((1,), jnp.float32))
        total = state.episode_returns + reward
        length = state.episode_lengths + 1
        state = state._replace(step_count=step, prev_dist_to_goal=distance,
                              episode_returns=total, episode_lengths=length)
        return state, reward, success, collision, timeout, boundary

    def _reset_crowd(self, rng):
        keys = jax.random.split(rng, 1)
        return jax.vmap(lambda key: self._reset_one(self.nav_cfg, key))(keys)
