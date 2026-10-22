"""Evaluate RL checkpoints or a CBF-QP controller and optionally render its episodes."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.jax_env.jax_render import render_jax_social_nav_frame


def _frame_to_rgb_array(frame):
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def save_video(path, frames, fps=30):
    import imageio_ffmpeg

    first = _frame_to_rgb_array(frames[0])
    height, width = first.shape[:2]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    # Allow Python's posix_spawn path rather than forking a running JAX process.
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, close_fds=False) as encoder:
        try:
            for frame in frames:
                rgb = _frame_to_rgb_array(frame)
                if rgb.shape != first.shape:
                    raise ValueError("Video frames must have identical dimensions")
                encoder.stdin.write(rgb.tobytes())
            encoder.stdin.close()
            error = encoder.stderr.read().decode("utf-8", errors="replace")
            if encoder.wait():
                raise RuntimeError(f"FFmpeg video encoding failed: {error}")
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise


def evaluate_controller(config, output, *, visualize=False, seeds=None, episodes=1):
    """Shared controller evaluation entry for Hydra and saved-run evaluation."""
    from trainer.platform import configure_jax_platform_for_device, validate_jax_backend_for_device

    platform = configure_jax_platform_for_device(config.device)
    import jax
    jax.config.update("jax_enable_x64", True)
    if platform != "auto":
        jax.config.update("jax_platforms", platform)
    validate_jax_backend_for_device(jax, config.device)
    from controller.simulate import execute

    return execute(config, output, visualize=visualize, seeds=seeds, episodes=episodes)


class Evaluator:
    def __init__(self, save_dir, visualize=False, device=None, seeds=None, episodes=None):
        self.save_dir = Path(save_dir)
        self.visualize = bool(visualize)
        self.seeds = seeds
        self.config = OmegaConf.load(self.save_dir / "config.yaml")
        self.is_controller = "controller" in self.config and "model" not in self.config
        self.episodes = int((1 if self.is_controller else 50) if episodes is None else episodes)
        backend = self.config.env.get("backend", "jax")
        if backend not in ("jax", "crax_mjx"):
            raise ValueError(f"Unsupported environment backend {backend!r}; use jax or crax_mjx")
        if self.visualize and self.config.env.get("backend", "jax") == "crax_mjx":
            # Select a renderer for this evaluation process before importing MuJoCo.
            os.environ.setdefault("MUJOCO_GL", "osmesa")
        if device:
            self.config.device = device
        # Device and float64 QP precision must be configured before importing JAX.
        from trainer.platform import configure_jax_platform_for_device
        configure_jax_platform_for_device(self.config.device)

    def evaluate_one_ckpt(self, checkpoint):
        import jax
        import jax.numpy as jnp
        from env.jax_env.jax_social_nav import config_from_omegaconf
        from model import get_jax_model_module
        from trainer.checkpoint import load_jax_checkpoint
        from trainer.jax_ppo_eval import evaluate_jax_env, _eval_seed_values

        checkpoint = Path(checkpoint)
        payload = load_jax_checkpoint(checkpoint)
        params = payload["params"]
        if self.config.env.get("backend", "jax") == "crax_mjx":
            from trainer.crax_eval import evaluate_crax_point
            video_dir = self.save_dir / f"visualize_{checkpoint.stem}" if self.visualize else None
            return evaluate_crax_point(self.config, params, episodes=self.episodes,
                                       seeds=self.seeds, video_dir=video_dir)
        module = get_jax_model_module(self.config.model)
        hparams = module.hparams_from_config(self.config.model, self.config.model.critic, act_dim=2)
        env_cfg = config_from_omegaconf(self.config)
        seed_values = _eval_seed_values(self.episodes, self.seeds)
        traces = [[] for _ in seed_values] if self.visualize else None

        def capture(state, history, _returns, success, collision, timeout, active):
            for index in np.flatnonzero(active):
                episode_state = jax.tree_util.tree_map(lambda value: np.asarray(value[index:index+1]).copy(), state)
                status = "collision" if collision[index] else "success" if success[index] else "timeout" if timeout[index] else "running"
                traces[index].append((episode_state, np.asarray(history[index:index+1]).copy(), status))

        metrics = evaluate_jax_env(
            params, hparams, env_cfg, episodes=self.episodes, seeds=self.seeds,
            policy_action_mean_fn=module.policy_action_mean,
            policy_action_to_env_action_fn=module.policy_action_to_env_action,
            include_paper_metrics=True,
            rollout_callback=capture if self.visualize else None,
        )
        if self.visualize:
            # Wait for ordered host callbacks before reading their captured states.
            jax.effects_barrier()
            video_dir = self.save_dir / f"visualize_{checkpoint.stem}"
            video_dir.mkdir(parents=True, exist_ok=True)
            for index, trace in enumerate(traces):
                frames = []
                trajectory = []
                for state, history, status in trace:
                    trajectory.append(state.robot_state[0, :2].copy())
                    qp_viz = None
                    if hasattr(module, "build_qp_context"):
                        context = module.build_qp_context(jnp.asarray(history.reshape((1, -1))), hparams)
                        qp_viz = {"topk_ids": np.asarray(context["qp_topk_ids"][0]),
                                  "lookahead_distance": getattr(hparams, "lookahead_distance", 0.0)}
                    frames.append(render_jax_social_nav_frame(env_cfg, state, status=status,
                                                             robot_traj=trajectory, qp_viz=qp_viz))
                status = trace[-1][-1]
                if status == "running":
                    status = "timeout"
                save_video(video_dir / f"episode_{index:04d}_seed_{int(seed_values[index])}_{status}.mp4",
                           frames, fps=int(round(1.0 / env_cfg.dt)))
            print(f"Videos: {video_dir}")
        return metrics

    def eval_all_ckpts(self, checkpoint=None):
        if self.is_controller:
            if checkpoint is not None:
                raise ValueError("CBF-QP controller evaluation does not use a checkpoint")
            return evaluate_controller(self.config, self.save_dir, visualize=self.visualize,
                                       seeds=self.seeds, episodes=self.episodes)
        paths = [Path(checkpoint)] if checkpoint else sorted(
            self.save_dir.glob("ckpt_*.pkl"), key=lambda path: int(path.stem.split("_")[-1]))
        if not paths:
            raise FileNotFoundError(f"No checkpoints in {self.save_dir}")
        results = {}
        for path in paths:
            results[path.name] = self.evaluate_one_ckpt(path)
            print(f"{path.name}: {json.dumps(results[path.name])}")
        aggregate = {"num_checkpoints": len(paths)}
        for key in ("mean_return", "success_rate", "collision_rate", "timeout_rate", "mean_min_clearance"):
            values = [metrics[key] for metrics in results.values() if metrics[key] is not None]
            aggregate[f"mean_{key}"] = float(np.mean(values)) if values else None
        results["aggregate"] = aggregate
        output = self.save_dir / "eval_results.json"
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Results: {output}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True, help="Run containing config.yaml (and checkpoints for RL)")
    parser.add_argument("--checkpoint", default="", help="One checkpoint; otherwise evaluate all")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds; RL defaults to 100,...,1000, controller to config.seed")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per base seed; default 50 for RL, 1 for controller")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--visualize", action="store_true", help="Render evaluation episodes as MP4 videos")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()] if args.seeds else None
    Evaluator(args.save_dir, args.visualize, args.device, seeds, args.episodes).eval_all_ckpts(args.checkpoint or None)
