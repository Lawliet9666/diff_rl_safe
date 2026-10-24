"""Uniform reset/observe/step over the original JAX crowd and reduced robot models."""
from typing import NamedTuple
import math

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from env.jax_env.jax_social_nav import (
    config_from_omegaconf, reset_env, step_env, compute_distances,
    navigation_done, _clip_robot_to_outer_bounds,
)
from env.obstacle.obstacle import advance_obstacles
from env.robots import make_robot
from env.robots.base import Obstacles


def quadrotor_from_crowd(crowd, robot_cfg):
    state = make_robot("quadrotor", robot_cfg).hover_state(
        jnp.array([crowd.robot_state[0, 0], crowd.robot_state[0, 1], robot_cfg.altitude]))
    yaw = crowd.robot_state[0, 2].astype(state.dtype)
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    R = jnp.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return state.at[6:15].set(R.reshape(-1))


def project_quadrotor(crowd, state):
    """Only x/y, world velocity and actual yaw enter the shared navigation state."""
    pose = make_robot("quadrotor").navigation_state(state)
    return crowd._replace(robot_state=pose[None, :],
                          robot_vel=state[None, 3:5],
                          robot_action=jnp.zeros_like(crowd.robot_action))


def project_humanoid(crowd, state):
    robot = make_robot("humanoid")
    return crowd._replace(robot_state=robot.navigation_state(state)[None, :],
                          robot_vel=robot.world_velocity(state)[None, :],
                          robot_action=jnp.zeros_like(crowd.robot_action))


def _finish_navigation_step(cfg, crowd, state, rng):
    """Original post-robot crowd update and termination; discard planar reward."""
    moved = advance_obstacles(cfg, crowd, rng)
    bounded = _clip_robot_to_outer_bounds(cfg, moved)
    correction = bounded.robot_state[0, :2]-state[:2]
    state = state.at[:2].set(bounded.robot_state[0, :2])
    dist, clearance = compute_distances(cfg, bounded)
    done, success, collision, timeout, next_step, violation = navigation_done(
        cfg, bounded, dist, clearance)
    lengths = bounded.episode_lengths+1
    bounded = bounded._replace(step_count=next_step, prev_dist_to_goal=dist.astype(jnp.float32),
                               episode_lengths=jnp.where(done, 0, lengths))
    metrics = dict(success=success, collision=collision, timeout=timeout,
        min_clearance=clearance, goal_distance=dist, boundary_violation=violation,
        episode_length=lengths, boundary_correction=correction)
    return state, bounded, done, metrics


def step_quadrotor(cfg, crowd, state, action, rng, robot_cfg, max_integration_dt, state_tolerance):
    """Replace robot propagation; retain robot-first crowd/boundary/terminal semantics.

    Shared navigation_done supplies termination flags without evaluating a planar reward.
    """
    count = max(1, math.ceil(cfg.dt/max_integration_dt))
    dt = cfg.dt/count
    robot = make_robot("quadrotor", robot_cfg)

    def advance(x, _):
        y = robot.step(x, action, dt)
        return y, y

    state, substeps = jax.lax.scan(advance, state, None, length=count)
    state, bounded, done, metrics = _finish_navigation_step(
        cfg, project_quadrotor(crowd, state), state, rng)
    domain = jnp.stack([substeps[:, 2]-robot_cfg.altitude_min,
        robot_cfg.altitude_max-substeps[:, 2], substeps[:, 14]-math.cos(robot_cfg.tilt_max),
        substeps[:, 15]-robot_cfg.thrust_min, robot_cfg.thrust_max-substeps[:, 15]])
    metrics["physical_state_limit"] = jnp.any(domain < -state_tolerance)
    return state, bounded, done, metrics


def step_humanoid(cfg, crowd, state, action, rng, robot, max_integration_dt, state_tolerance):
    count = max(1, math.ceil(cfg.dt/max_integration_dt))
    dt = cfg.dt/count

    def advance(x, _):
        y = robot.step(x, action, dt)
        return y, robot.velocity_margin(y)

    state, margins = jax.lax.scan(advance, state, None, length=count)
    state, crowd, done, info = _finish_navigation_step(
        cfg, project_humanoid(crowd, state), state, rng)
    info["physical_state_limit"] = jnp.any(margins < -state_tolerance)
    return state, crowd, done, info


class NavigationState(NamedTuple):
    robot: jax.Array
    crowd: object


class NavigationObservation(NamedTuple):
    robot: jax.Array
    goal: jax.Array
    obstacles: Obstacles


class NavigationEnv:
    """Single-episode environment interface for the supported robot models.

    The original planar environment handles unicycle rewards and transitions.
    Quadrotor/humanoid transitions preserve navigation flags and return reward=None.
    """

    def __init__(self, config, *, state_tolerance):
        if config.env.get("backend", "jax") != "jax":
            raise ValueError("NavigationEnv requires the JAX crowd backend")
        self.robot_type = str(config.robot.type)
        if self.robot_type not in {"unicycle", "quadrotor", "humanoid"}:
            raise ValueError("NavigationEnv supports unicycle, quadrotor and humanoid")
        self.robot_config = OmegaConf.create(OmegaConf.to_container(config.robot, resolve=True))
        self.cfg = config_from_omegaconf(config)
        self.robot = self.cfg.robot
        self.qp_obs_top_k = config.controller.get("qp_obs_top_k")
        if self.qp_obs_top_k is not None and (
                not isinstance(self.qp_obs_top_k, int) or isinstance(self.qp_obs_top_k, bool)
                or self.qp_obs_top_k <= 0):
            raise ValueError("controller.qp_obs_top_k must be null or a positive integer")
        self.state_tolerance = float(state_tolerance)
        if not math.isfinite(self.state_tolerance) or self.state_tolerance <= 0:
            raise ValueError("state_tolerance must be positive")
        self.integration_dt = self.cfg.dt
        if self.robot_type in {"quadrotor", "humanoid"}:
            self.max_integration_dt = float(self.robot_config.max_integration_dt)
            if not math.isfinite(self.max_integration_dt) or self.max_integration_dt <= 0:
                raise ValueError("robot.max_integration_dt must be positive")
            self.integration_dt /= max(1, math.ceil(self.cfg.dt/self.max_integration_dt))
        self._advance = jax.jit(self._step)

    def reset(self, rng):
        crowd = reset_env(self.cfg, rng, 1)
        if self.robot_type == "quadrotor":
            state = quadrotor_from_crowd(crowd, self.robot_config)
            crowd = project_quadrotor(crowd, state)
        elif self.robot_type == "humanoid":
            state = jnp.concatenate([crowd.robot_state[0].astype(jnp.float64), jnp.zeros(2)])
            crowd = project_humanoid(crowd, state)
        else:
            state = crowd.robot_state[0].astype(jnp.float64)
        return NavigationState(state, crowd)

    def observe(self, state):
        crowd = state.crowd
        mask = crowd.human_mask[0]
        if self.qp_obs_top_k is not None:
            distance = jnp.linalg.norm(crowd.human_positions[0]-state.robot[:2], axis=1)
            _, indices = jax.lax.top_k(
                -jnp.where(mask > .5, distance, jnp.inf),
                min(self.qp_obs_top_k, self.cfg.num_humans))
            # Keep scene indices and fixed shapes; only QP visibility changes.
            mask = jnp.zeros_like(mask).at[indices].set(mask[indices])
        obstacles = Obstacles(
            crowd.human_positions[0].astype(jnp.float64),
            crowd.human_vels[0].astype(jnp.float64),
            crowd.human_radii[0].astype(jnp.float64), mask)
        return NavigationObservation(
            state.robot, crowd.goal_pos[0].astype(jnp.float64), obstacles)

    def step(self, state, action, rng):
        """Return (state, reward, done, info), with the applied action in info."""
        return self._advance(state, action, rng)

    def _step(self, state, action, rng):
        if self.robot_type == "quadrotor":
            robot, crowd, done, info = step_quadrotor(
                self.cfg, state.crowd, state.robot, action, rng,
                self.robot_config, self.max_integration_dt, self.state_tolerance)
            reward = None
            applied = action
        elif self.robot_type == "humanoid":
            robot, crowd, done, info = step_humanoid(
                self.cfg, state.crowd, state.robot, action, rng,
                self.robot, self.max_integration_dt, self.state_tolerance)
            reward = None
            applied = action
        else:
            crowd, reward, done, info = step_env(
                self.cfg, state.crowd, action[None, :], rng)
            robot = crowd.robot_state[0].astype(jnp.float64)
            applied = crowd.robot_action[0]
            info = {**info, "physical_state_limit": jnp.array(False)}
        return (NavigationState(robot, crowd), reward, done,
                {**info, "applied_action": applied})
