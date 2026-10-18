"""Compiled deterministic evaluation and rendering of exactly the measured states."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from model import get_jax_model_module
from trainer.jax_ppo_eval import _eval_seed_values, _paper_metric_summary, _safe_mean_std
from trainer.jax_ppo_rollout import _select_done
from trainer.jax_episode import episode_outcomes


def _make_evaluation(env, module, hp, record):
    use_qp = hasattr(module, "build_qp_context")

    def run(params, keys):
        state = jax.vmap(env.reset)(keys)
        frame = jax.vmap(env.observe)(state)
        history = jnp.repeat(frame[:, None], hp.history_len, axis=1)
        batch_size = frame.shape[0]
        active = jnp.ones(batch_size, bool)
        initial = {"qpos": state.physics.qpos, "qvel": state.physics.qvel,
                   "mocap_pos": state.physics.mocap_pos, "mask": state.crowd.human_mask[:, 0]}

        def body(carry, _):
            state, history, active = carry
            obs = history.reshape(batch_size, -1)
            if use_qp:
                context = module.build_qp_context(obs, hp)
                mean = module.policy_action_mean(params, hp, obs, qp_context=context)
                ids = context["qp_topk_ids"]
                # Diagnostics use zero policy noise; action remains the historical
                # deterministic mean/action mapping used by checkpoint evaluation.
                *_, diagnostics = module.get_action_and_value(params, hp, obs,
                    noise=jnp.zeros((batch_size, 2), jnp.float32),
                    return_diagnostics=True, qp_context=context)
            else:
                mean = module.policy_action_mean(params, hp, obs)
                ids = jnp.zeros((batch_size, 0), jnp.int32)
                diagnostics = {}
            action = module.policy_action_to_env_action(params, mean)
            proposed, reward, done, metrics = jax.vmap(env.step)(state, action)
            next_state = jax.tree_util.tree_map(lambda a,b: _select_done(active, b, a), state, proposed)
            next_frame = jax.vmap(env.observe)(next_state)
            next_history = jnp.concatenate([history[:, 1:], next_frame[:, None]], axis=1)
            out = {name: jnp.where(active, value, 0.) for name, value in metrics.items()}
            outcomes = episode_outcomes(active & (done > .5), metrics["success"], metrics["collision"])
            out.update({name: value.astype(jnp.float32)
                        for name, value in zip(("success", "collision", "timeout"), outcomes)})
            out.update(reward=jnp.where(active, reward, 0.), active=active, qp_ids=ids)
            out["min_clearance"] = jnp.where(active, metrics["min_clearance"], jnp.inf)
            for name in ("qp_slack", "alpha", "beta"):
                if name in diagnostics:
                    value = diagnostics[name]
                    if value.ndim > 1:
                        masks = context["qp_obs"][..., 6:].reshape(value.shape + (6,))[..., 5]
                        value = jnp.sum(value * masks, axis=-1) / jnp.maximum(masks.sum(axis=-1), 1.)
                    out[name] = jnp.where(active, value, 0.)
            if record:
                out.update(qpos=next_state.physics.qpos, qvel=next_state.physics.qvel,
                           mocap_pos=next_state.physics.mocap_pos)
            return (next_state, next_history, active & (done < .5)), out

        _, trace = jax.lax.scan(body, (state, history, active), None, length=env.nav_cfg.max_steps)
        return trace, initial if record else {}

    return jax.jit(run)


def evaluate_crax_point(config, params, episodes=1, seeds=None, video_dir=None, env=None):
    if env is None:
        from env.crax_point_crowd import CraxPointCrowd
        env = CraxPointCrowd(config)
    module = get_jax_model_module(config.model)
    hp = module.hparams_from_config(config.model, config.model.critic, act_dim=2)
    seed_values = _eval_seed_values(episodes, seeds)
    record = video_dir is not None
    cache = getattr(env, "_evaluation_cache", {})
    cache_key = (str(config.model.type), len(seed_values), record)
    if cache_key not in cache:
        cache[cache_key] = _make_evaluation(env, module, hp, record)
        env._evaluation_cache = cache
    trace, initial = jax.device_get(cache[cache_key](params, jax.vmap(jax.random.PRNGKey)(seed_values)))
    records = []
    for index, seed in enumerate(seed_values):
        length = int(trace["active"][:, index].sum())
        r = dict(seed=int(seed), episode_length=length,
            **{name: float(trace[name][:, index].sum()) for name in ("success", "collision", "timeout", "cost")})
        # Match the JAX evaluator if the evaluation horizon itself ends an episode.
        if not any(r[name] for name in ("success", "collision", "timeout")):
            r["timeout"] = 1.0
        r["return"] = float(trace["reward"][:, index].sum())
        clearance = float(trace["min_clearance"][:, index].min())
        r["min_clearance"] = clearance if np.isfinite(clearance) else None
        r["physical_contact"] = float(trace["physical_contact"][:, index].max())
        for name in ("tracking_speed_error", "tracking_omega_error", "tracking_lateral_speed",
                     "actuator_saturation", "qp_slack", "alpha", "beta"):
            if name in trace:
                r[name] = float(trace[name][:length, index].mean())
        records.append(r)
    if record:
        from env.crax_render import RecordedPointRenderer
        from scripts.eval import save_video
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        renderer = RecordedPointRenderer(env)
        try:
            for i, r in enumerate(records):
                length = r["episode_length"]
                qpos = np.concatenate([initial["qpos"][i:i+1], trace["qpos"][:length, i]])
                qvel = np.concatenate([initial["qvel"][i:i+1], trace["qvel"][:length, i]])
                mocap = np.concatenate([initial["mocap_pos"][i:i+1], trace["mocap_pos"][:length, i]])
                totals = np.r_[0., np.cumsum(trace["reward"][:length, i])]
                status = "collision" if r["collision"] else "success" if r["success"] else "timeout"
                frames = [renderer.frame(qpos[t], qvel[t], mocap[t], initial["mask"][i],
                    trace["qp_ids"][max(t-1, 0), i], qpos[:t+1, :2], totals[t], t,
                    status if t == length else "running") for t in range(length+1)]
                filename = f"episode_{i:03d}_seed_{r['seed']}.mp4"
                save_video(video_dir / filename, frames, fps=1./env.nav_cfg.dt)
                r["video"] = filename
                np.savez_compressed(video_dir / filename.replace('.mp4', '.npz'),
                    qpos=qpos, qvel=qvel, mocap_pos=mocap, reward=trace["reward"][:length, i],
                    qp_ids=trace["qp_ids"][:length, i], mask=initial["mask"][i], seed=r["seed"])
                print(f"Video: {video_dir / filename}", flush=True)
        finally:
            renderer.close()
        (video_dir / "episodes.json").write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    clearances = np.array([np.nan if r["min_clearance"] is None else r["min_clearance"] for r in records])
    mean_clearance, std_clearance = _safe_mean_std(clearances)
    metrics = {"mean_return": float(np.mean([r["return"] for r in records])),
        "std_return": float(np.std([r["return"] for r in records])),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "timeout_rate": float(np.mean([r["timeout"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_min_clearance": mean_clearance, "std_min_clearance": std_clearance,
        "mean_cost": float(np.mean([r["cost"] for r in records])),
        "physical_contact_rate": float(np.mean([r["physical_contact"] for r in records]))}
    for name in ("tracking_speed_error", "tracking_omega_error", "tracking_lateral_speed",
                 "actuator_saturation", "qp_slack", "alpha", "beta", "cost", "physical_contact"):
        if name in trace:
            metrics[name] = float(trace[name].sum() / max(trace["active"].sum(), 1))
    metrics.update(_paper_metric_summary(np.array([r["success"] for r in records]),
        np.array([r["episode_length"] for r in records]), clearances, env.nav_cfg.dt))
    return metrics
