"""Measure complete PPO updates, excluding evaluation, with synchronized timing."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import os

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["crax"], default="crax")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--method", default="barriernet_gnn")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--name", default="crax_benchmark")
    args = parser.parse_args()
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
    OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)
    env_name = "crax_point_crowd"
    nenv = args.num_envs or 128
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/"config")):
        config = compose(config_name="config", overrides=[f"env={env_name}", "robot=point",
            f"model={args.method}", "device=cuda", f"run_name={args.name}",
            "save_dir=outputs/crax_validation/runs", f"trainer.num_envs={nenv}",
            f"trainer.timesteps_per_batch={args.batch}",
            f"trainer.total_timesteps={args.batch*args.updates}"])
    from scripts.run_ppo_base import build_trainer
    trainer = build_trainer(config)
    import jax
    import jax.numpy as jnp
    import wandb
    from trainer.jax_checkpoint import save_jax_checkpoint
    state = trainer.train_state
    env_state = trainer._reset_training_env()
    jax.block_until_ready(env_state)
    state = trainer.train_state
    params, opt_state, rng, global_step = state.params, state.opt_state, state.rng, jnp.int32(0)
    scalar_args = [jnp.float32(config.trainer.lr), jnp.float32(config.trainer.clip),
        jnp.float32(config.trainer.ent_coef), jnp.float32(config.trainer.max_grad_norm),
        jnp.float32(config.trainer.vf_coef), jnp.float32(config.trainer.value_clip), jnp.bool_(True)]
    fn = trainer.jitted_train_step
    t = time.perf_counter()
    fn = fn.lower(params, opt_state, env_state, rng, global_step, *scalar_args).compile()
    compile_seconds = time.perf_counter()-t
    print(f"compile_seconds={compile_seconds:.3f}", flush=True)
    durations = []
    epochs_completed = []
    for update in range(args.updates):
        t = time.perf_counter()
        params, opt_state, env_state, rng, global_step, pre, post = fn(
            params, opt_state, env_state, rng, global_step, *scalar_args)
        jax.block_until_ready((params, opt_state, global_step, pre, post))
        durations.append(time.perf_counter()-t)
        epochs_completed.append(float(post["metrics"]["epochs_completed"]))
        update_record = {"step": int(global_step), "seconds": durations[-1],
            "ppo": {key: float(value) for key, value in jax.device_get(post["metrics"]).items()},
            "rollout": trainer._train_metrics_from_host_summary(jax.device_get(pre))}
        with (trainer.save_dir/"benchmark_updates.jsonl").open("a") as handle:
            handle.write(json.dumps(update_record)+'\n')
        trainer._replace_train_state(params=params, opt_state=opt_state, env_state=env_state,
                                     rng=rng, global_step=int(global_step))
        metrics = trainer._train_metrics_from_host_summary(jax.device_get(pre))
        if not bool(pre["gae_finite"]) or not bool(post["params_finite_after_update"]):
            raise FloatingPointError("Non-finite benchmark update")
        print(f"update={update+1} seconds={durations[-1]:.3f} steps={int(global_step)} metrics={metrics}", flush=True)
    memory = jax.devices()[0].memory_stats() or {}
    warm = float(sum(durations[1:])/max(len(durations)-1, 1)) if len(durations)>1 else durations[0]
    report = dict(backend=args.backend, method=args.method, num_envs=nenv, batch=args.batch,
        compile_seconds=compile_seconds, update_seconds=durations,
        epochs_completed=epochs_completed, learning_rate=float(config.trainer.lr),
        excludes_evaluation_and_reset_compilation=True,
        warmed_update_seconds=warm, transitions_per_second=args.batch/warm,
        estimated_20m_seconds=20_000_000/args.batch*warm,
        memory_bytes=memory, jax=jax.__version__)
    (trainer.save_dir/"benchmark.json").write_text(json.dumps(report, indent=2)+'\n')
    save_jax_checkpoint(trainer.save_dir, params, step=int(global_step), performance=0., max_keep=10)
    wandb.finish()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
