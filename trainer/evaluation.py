"""Shared deterministic navigation evaluation."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model import make_action_fn
from env.robots.base import Obstacles
from controller.cbf_qp import _nearest_obstacle_ids


DEFAULT_EVAL_SEEDS = tuple(range(100, 1001, 100))


def episode_outcomes(done, success, collision, qp_failure=None):
    """Mutually exclusive terminal reasons; rejected QPs are not timeouts."""
    ended = jnp.asarray(done) > 0.5
    failed = ended & (jnp.zeros_like(ended) if qp_failure is None else jnp.asarray(qp_failure) > .5)
    collision = ended & (jnp.asarray(collision) > 0.5) & ~failed
    success = ended & (jnp.asarray(success) > 0.5) & ~collision
    success &= ~failed
    timeout = ended & ~(success | collision | failed)
    return success, collision, timeout, failed


def eval_seed_values(episodes, seeds=None):
    base_seeds = DEFAULT_EVAL_SEEDS if seeds is None else tuple(int(seed) for seed in seeds)
    if not base_seeds:
        raise ValueError("at least one evaluation seed is required")
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    return np.asarray([seed + episode for seed in base_seeds
                       for episode in range(episodes)], dtype=np.uint32)


def safe_mean_std(values):
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def paper_metric_summary(success, lengths, clearance, dt):
    selected = np.asarray(success) > 0.5
    times = np.asarray(lengths).astype(np.float32) * float(dt)
    time_mean, time_std = safe_mean_std(times[selected])
    dist_mean, dist_std = safe_mean_std(np.asarray(clearance)[selected])
    return dict(traj_time=time_mean, std_traj_time=time_std,
                min_dist=dist_mean, std_min_dist=dist_std)


def _select_active(active, old_tree, new_tree):
    def select(old, new):
        if new.ndim == 0:
            return new
        shape = (active.shape[0],) + (1,) * (new.ndim - 1)
        return jnp.where(active.reshape(shape), new, old)
    return jax.tree_util.tree_map(select, old_tree, new_tree)


def _make_rollout(env, module, hp, cfg, capture, safety_filter=None):
    use_qp = hasattr(module, "build_qp_context")
    if safety_filter is not None and use_qp:
        raise ValueError("external safety filter cannot wrap a policy with an internal QP")
    step_fn, observe_fn = env.step_fn, env.observe_fn

    def transition(params, state, history, key, active):
        obs = history.reshape(history.shape[0], -1)
        if use_qp:
            # Completed episodes must not keep an infeasible QP in the live batch.
            obs = jnp.where(active[:, None], obs, 0.)
            context_kwargs = {}
            if getattr(hp, "robot_state_dim", 0):
                initial = hp.robot.initial_state(jnp.zeros(3, state.robot_state.dtype))
                context_kwargs["robot_state"] = jnp.where(active[:, None], state.robot_state, initial)
            context = module.build_qp_context(obs, hp, **context_kwargs)
            # Evaluate the deterministic policy mean.
            action, *_, diagnostics = module.get_action_and_value(
                params, hp, obs, noise=jnp.zeros((obs.shape[0], hp.act_dim), jnp.float32),
                return_diagnostics=True, qp_context=context)
            ids = context["qp_topk_ids"]
        else:
            action_fn = make_action_fn(params, hp, module.policy_action_mean,
                                       module.policy_action_to_env_action)
            action, diagnostics = action_fn(obs)
            ids = jnp.zeros((obs.shape[0], 0), jnp.int32)
        qp_failure = (diagnostics.get("qp_failure", jnp.zeros_like(active))
                      if use_qp else jnp.zeros_like(active))
        if safety_filter is not None:
            obstacles = Obstacles(state.human_positions, state.human_vels,
                                  state.human_radii, state.human_mask)
            action, diagnostics = safety_filter(state.robot_state, action, obstacles, active=active,
                                                 allow_failure=True)
            qp_failure |= diagnostics["qp_failure"]
            ids = diagnostics['qp_ids']
        qp_failure &= active
        action = jnp.where(qp_failure[:, None], 0., action)
        step_kwargs = {"qp_failure": qp_failure} if use_qp or safety_filter is not None else {}
        proposed, reward, done, metrics = step_fn(cfg, state, action, key, **step_kwargs)
        info = {name: metrics[name] for name in ("success", "collision", "min_clearance")}
        info["qp_failure"] = qp_failure.astype(jnp.float32)
        if "physical_state_limit" in metrics:
            info["physical_state_limit"] = metrics["physical_state_limit"].astype(jnp.float32)
        if "physical_contact" in metrics:
            info["physical_contact"] = metrics["physical_contact"]
        for name in ("qp_slack", "alpha", "beta"):
            if name in diagnostics:
                value = diagnostics[name]
                if value.ndim > 1:
                    mask = context["qp_obs"][..., 6:].reshape((value.shape[0], -1, 6))[..., 5]
                    if value.ndim == 3:
                        mask = jnp.broadcast_to(mask[..., None], value.shape)
                    axes = tuple(range(1, value.ndim))
                    value = jnp.sum(value * mask, axis=axes) / jnp.maximum(mask.sum(axis=axes), 1.)
                info[name] = value
        info.update({name: value for name, value in diagnostics.items() if name.startswith('filter/')})
        return proposed, reward, done, info, ids

    def run(params, state, history):
        size = history.shape[0]
        zero = jnp.zeros(size, jnp.float32)
        false = jnp.zeros(size, bool)
        key = jax.random.PRNGKey(0)
        info_shapes = jax.eval_shape(transition, params, state, history, key, ~false)[3]
        extra_names = tuple(name for name in info_shapes
                            if name not in {"success", "collision", "min_clearance", "qp_failure"})
        initial = dict(state=state, history=history, done=false, returns=zero,
                       success=false, collision=false, timeout=false, qp_failure=false,
                       lengths=jnp.zeros(size, jnp.int32),
                       min_clearance=jnp.full(size, jnp.inf, jnp.float32),
                       extras={name: jnp.zeros(size, info_shapes[name].dtype) for name in extra_names},
                       contact_max=zero, key=key, step=jnp.asarray(0, jnp.int32))

        if capture is not None:
            ids = (module.build_qp_context(history.reshape(size, -1), hp, **(dict(robot_state=state.robot_state) if getattr(hp, "robot_state_dim", 0) else {}))["qp_topk_ids"]
                   if use_qp else jnp.zeros((size, 0), jnp.int32))
            if (safety_filter is not None
                    and safety_filter.keywords.get("constraint_mode", "cbf") != "physical"):
                obstacles = Obstacles(state.human_positions, state.human_vels,
                                      state.human_radii, state.human_mask)
                ids = _nearest_obstacle_ids(state.robot_state[..., :2], obstacles,
                                            safety_filter.keywords["qp_obs_top_k"])
            jax.debug.callback(capture, env.snapshot_fn(state), history, zero,
                               false, false, false, false, ~false, ids, zero, ordered=True)

        def cond(carry):
            return (carry["step"] < cfg.max_steps) & (~jnp.all(carry["done"]))

        def body(carry):
            key, step_key = jax.random.split(carry["key"])
            proposed, reward, done, info, ids = transition(
                params, carry["state"], carry["history"], step_key, ~carry["done"])
            active = ~carry["done"]
            success, collision, timeout, qp_failure = episode_outcomes(
                active & (done > .5), info["success"], info["collision"], info["qp_failure"])
            state = _select_active(active, carry["state"], proposed)
            frame = observe_fn(cfg, state, hp)
            shifted = jnp.concatenate([carry["history"][:, 1:], frame[:, None]], axis=1)
            history = jnp.where(active[:, None, None], shifted, carry["history"])
            returns = carry["returns"] + jnp.where(active, reward, 0.)
            contact = (jnp.maximum(carry["contact_max"],
                                  jnp.where(active, info["physical_contact"], 0.))
                       if "physical_contact" in info else carry["contact_max"])
            if capture is not None:
                jax.debug.callback(capture, env.snapshot_fn(state), history, returns,
                                   success, collision, timeout, qp_failure, active, ids,
                                   jnp.where(active, reward, 0.), ordered=True)
            return dict(
                state=state, history=history, done=carry["done"] | (active & (done > .5)),
                returns=returns, success=carry["success"] | success,
                collision=carry["collision"] | collision, timeout=carry["timeout"] | timeout,
                qp_failure=carry["qp_failure"] | qp_failure,
                lengths=carry["lengths"] + active.astype(jnp.int32),
                min_clearance=jnp.minimum(carry["min_clearance"],
                                         jnp.where(active, info["min_clearance"], jnp.inf)),
                extras={name: (jnp.maximum(carry["extras"][name], jnp.where(active, info[name], 0.))
                               if name.startswith('filter/') and name.endswith('_max') else
                               carry["extras"][name] + jnp.where(active, info[name], 0.))
                        for name in extra_names},
                contact_max=contact, key=key, step=carry["step"] + 1)

        result = jax.lax.while_loop(cond, body, initial)
        result["timeout"] |= ~result["done"] & ~result["success"] & ~result["collision"] & ~result["qp_failure"]
        return {name: result[name] for name in (
            "returns", "success", "collision", "timeout", "qp_failure", "lengths",
            "min_clearance", "extras", "contact_max")}

    return jax.jit(run)


def _summarize(result, seeds, dt):
    returns, lengths = result["returns"], result["lengths"]
    clearance = result["min_clearance"]
    mean_clearance, std_clearance = safe_mean_std(clearance)
    metrics = dict(mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                   success_rate=float(np.mean(result["success"])),
                   collision_rate=float(np.mean(result["collision"])),
                   timeout_rate=float(np.mean(result["timeout"])),
                   qp_failure_rate=float(np.mean(result["qp_failure"])),
                   mean_episode_length=float(np.mean(lengths)),
                   mean_min_clearance=mean_clearance, std_min_clearance=std_clearance)
    metrics.update(paper_metric_summary(result["success"], lengths, clearance, dt))
    total_steps = max(int(lengths.sum()), 1)
    metrics.update({name: float(value.max() if name.startswith('filter/') and name.endswith('_max')
                                else value.sum() / total_steps)
                    for name, value in result["extras"].items()})
    if "physical_contact" in result["extras"]:
        metrics["physical_contact_rate"] = float(np.mean(result["contact_max"]))
    records = []
    for index, seed in enumerate(seeds):
        record = dict(seed=int(seed), episode_length=int(lengths[index]),
                      **{name: float(result[name][index])
                         for name in ("success", "collision", "timeout", "qp_failure")})
        record["return"] = float(returns[index])
        record["min_clearance"] = float(clearance[index]) if np.isfinite(clearance[index]) else None
        record.update({name: float(values[index] if name.startswith('filter/') and name.endswith('_max')
                                   else values[index] / max(int(lengths[index]), 1))
                       for name, values in result["extras"].items()})
        if "physical_contact" in result["extras"]:
            record["physical_contact"] = float(result["contact_max"][index])
        records.append(record)
    return metrics, records


def evaluate_env(env, params, hparams, env_cfg, model_module, *,
                 episodes=1, seeds=None, video_dir=None, rollout_callback=None, safety_filter=None):
    """Evaluate fixed-seed episodes without resetting completed environments."""
    seeds = eval_seed_values(episodes, seeds)
    traces = [[] for _ in seeds] if video_dir is not None else None

    def capture(snapshot, history, returns, success, collision, timeout, qp_failure, active, ids, reward):
        if rollout_callback is not None:
            rollout_callback(snapshot, history, returns, success, collision, timeout, active)
        if traces is not None:
            for index in np.flatnonzero(active):
                status = ("qp_failure" if qp_failure[index] else "collision" if collision[index] else "success" if success[index]
                          else "timeout" if timeout[index] else "running")
                traces[index].append(dict(
                    snapshot=jax.tree_util.tree_map(
                        lambda value: np.asarray(value[index:index+1]).copy(), snapshot),
                    history=np.asarray(history[index:index+1]).copy(),
                    returns=float(returns[index]), reward=float(reward[index]), status=status,
                    qp_ids=np.asarray(ids[index]).copy()))

    callback = capture if traces is not None or rollout_callback is not None else None
    if callback is None:
        cache = getattr(env, "_evaluation_cache", {})
        cache_key = (model_module.__name__, repr(hparams), repr(env_cfg), repr(safety_filter))
        if cache_key not in cache:
            cache[cache_key] = _make_rollout(env, model_module, hparams, env_cfg, None, safety_filter)
            env._evaluation_cache = cache
        run = cache[cache_key]
    else:
        run = _make_rollout(env, model_module, hparams, env_cfg, callback, safety_filter)
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    state = env.reset_fn(env_cfg, keys, len(seeds))
    frame = env.observe_fn(env_cfg, state, hparams)
    history = jnp.repeat(frame[:, None, :], hparams.history_len, axis=1)
    result = jax.device_get(run(params, state, history))
    if callback is not None:
        jax.effects_barrier()
    metrics, records = _summarize(result, seeds, env_cfg.dt)
    if video_dir is not None:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env.save_videos(video_dir, traces, records, env_cfg, hparams, model_module)
    return metrics
