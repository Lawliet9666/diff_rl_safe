"""Evaluate checkpoints and optionally render the very same evaluation episodes."""

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


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def render_jax_social_nav_frame(
    env_cfg,
    state,
    status="",
    size=740,
    robot_traj=None,
    qp_viz=None,
):
    """Render one JAX SocialNav state as a Matplotlib frame."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    robot_state = np.asarray(state.robot_state[0], dtype=float)
    robot_pos = robot_state[:2]
    robot_theta = float(robot_state[2]) if robot_state.shape[0] > 2 else 0.0
    robot_radius = float(env_cfg.robot_radius)
    human_positions = np.asarray(state.human_positions[0], dtype=float)
    human_radii = np.asarray(state.human_radii[0], dtype=float)
    human_mask = np.asarray(state.human_mask[0], dtype=float) > 0.5
    goal_pos = np.asarray(state.goal_pos[0], dtype=float)

    robot_traj_arr = np.asarray(robot_traj if robot_traj is not None else [robot_pos], dtype=float)
    start_pos = np.asarray(robot_traj_arr[0] if robot_traj_arr.size else robot_pos, dtype=float)

    low = np.asarray(getattr(env_cfg, "workspace_low", [-6.0, -6.0]), dtype=float)
    high = np.asarray(getattr(env_cfg, "workspace_high", [6.0, 6.0]), dtype=float)
    boundary_buffer = max(float(getattr(env_cfg, "boundary_buffer", 0.0)), 0.0)
    plot_low = low - boundary_buffer
    plot_high = high + boundary_buffer
    plot_span = np.maximum(plot_high - plot_low, 1e-6)
    plot_aspect = float(plot_span[0] / plot_span[1])

    frame_width = max(16, int(np.ceil(float(size) / 16.0)) * 16)
    frame_height = max(16, int(np.ceil((frame_width / plot_aspect) / 16.0)) * 16)
    dpi = max(80, int(size / 7.4))
    fig = Figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.90)

    ax.clear()
    ax.set_xlim(float(plot_low[0]), float(plot_high[0]))
    ax.set_ylim(float(plot_low[1]), float(plot_high[1]))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    robot_color = "#3b82f6"
    traj_cmap = LinearSegmentedColormap.from_list("robot_traj", ["#dbeafe", robot_color])
    qp_viz = qp_viz or {}
    robot_type = str(getattr(env_cfg, "robot_type", "unicycle"))
    lookahead_distance = float(qp_viz.get("lookahead_distance", 0.0) or 0.0)
    if robot_type != "unicycle":
        lookahead_distance = 0.0

    topk_ids = np.asarray(qp_viz.get("topk_ids", []), dtype=int).reshape(-1)
    human_colors = np.repeat(np.asarray(to_rgba("#8c8c8c"))[None, :], len(human_positions), axis=0)
    for human_id in topk_ids:
        if 0 <= human_id < len(human_positions) and human_mask[human_id]:
            human_colors[human_id] = to_rgba("#dc2626")

    for i, pos in enumerate(human_positions):
        if i < human_mask.shape[0] and not human_mask[i]:
            continue
        ax.add_artist(Circle(pos, human_radii[i], facecolor=human_colors[i], edgecolor="none", linewidth=0.0, alpha=0.95, zorder=2))

    if robot_traj_arr.shape[0] > 1:
        points = robot_traj_arr.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_values = np.linspace(0.0, 1.0, len(segments), dtype=float)
        collection = LineCollection(
            segments,
            cmap=traj_cmap,
            norm=Normalize(vmin=0.0, vmax=1.0),
            linewidths=4.4,
            alpha=0.98,
            zorder=4,
        )
        collection.set_array(seg_values)
        collection.set_capstyle("round")
        ax.add_collection(collection)

    ax.add_artist(Circle(robot_pos, robot_radius, facecolor=robot_color, edgecolor="none", linewidth=0.0, alpha=0.82, zorder=6))
    if lookahead_distance > 0.0:
        lookahead_point = robot_pos + lookahead_distance * np.asarray([np.cos(robot_theta), np.sin(robot_theta)], dtype=float)
        ax.add_artist(
            Circle(
                lookahead_point,
                robot_radius + lookahead_distance,
                facecolor="none",
                edgecolor="#1d4ed8",
                linewidth=1.4,
                linestyle="--",
                alpha=0.7,
                zorder=7,
            )
        )
        ax.plot([robot_pos[0], lookahead_point[0]], [robot_pos[1], lookahead_point[1]], color="#1d4ed8", linewidth=1.2, alpha=0.85, zorder=8)
        ax.scatter([lookahead_point[0]], [lookahead_point[1]], s=56, c="#1d4ed8", edgecolors="white", linewidths=0.8, zorder=9)

    ax.scatter([start_pos[0]], [start_pos[1]], s=420, marker="^", c="#ead94c", edgecolors="none", linewidths=0.0, zorder=8)
    ax.add_artist(Circle(goal_pos, 0.3, facecolor="#ead94c", edgecolor="#d2bc38", linewidth=1.2, alpha=0.65, zorder=7))

    start_handle = Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="none", markersize=9, label="Start")
    goal_handle = Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="#d2bc38", markeredgewidth=1.0, markersize=9, label="Goal")
    traj_handle = Line2D([0], [0], color=robot_color, linewidth=4.4, alpha=0.98, label="Trajectory")
    handles = [start_handle, goal_handle, traj_handle]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=13, borderpad=0.3, handletextpad=0.4)

    ax.grid(False)
    ax.tick_params(labelsize=18, colors="black")
    current_time = float(_scalar(state.step_count[0])) * float(env_cfg.dt)
    title = f"Time: {current_time:.1f}s"
    if status:
        title += f" | {status}"
    ax.set_title(title, fontsize=20, pad=7, color="black")

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


class Evaluator:
    def __init__(self, save_dir, visualize=False, device=None, seeds=None, episodes=50):
        self.save_dir = Path(save_dir)
        self.visualize = bool(visualize)
        self.seeds = seeds
        self.episodes = int(episodes)
        self.config = OmegaConf.load(self.save_dir / "config.yaml")
        backend = self.config.env.get("backend", "jax")
        if backend not in ("jax", "crax_mjx"):
            raise ValueError(f"Unsupported environment backend {backend!r}; use jax or crax_mjx")
        if self.visualize and self.config.env.get("backend", "jax") == "crax_mjx":
            # Select a renderer for this evaluation process before importing MuJoCo.
            os.environ.setdefault("MUJOCO_GL", "osmesa")
        if device:
            self.config.device = device
        # Device and float64 QP precision must be configured before importing JAX.
        from trainer.jax_platform import configure_jax_platform_for_device
        configure_jax_platform_for_device(self.config.device)

    def evaluate_one_ckpt(self, checkpoint):
        import jax
        import jax.numpy as jnp
        from env.jax_social_nav import config_from_omegaconf
        from model import get_jax_model_module
        from trainer.jax_checkpoint import load_jax_checkpoint
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
    parser.add_argument("--save-dir", required=True, help="Run containing config.yaml and checkpoints")
    parser.add_argument("--checkpoint", default="", help="One checkpoint; otherwise evaluate all")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds; default 100,200,...,1000")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per base seed")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--visualize", action="store_true", help="Render evaluation episodes as MP4 videos")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()] if args.seeds else None
    Evaluator(args.save_dir, args.visualize, args.device, seeds, args.episodes).eval_all_ckpts(args.checkpoint or None)
