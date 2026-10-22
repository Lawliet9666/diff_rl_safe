"""Original two-dimensional JAX SocialNav renderer."""
import numpy as np


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


def save_videos(video_dir, traces, records, env_cfg, hparams, module):
    import jax.numpy as jnp
    from scripts.eval import save_video

    for index, (trace, record) in enumerate(zip(traces, records)):
        trajectory, frames = [], []
        for frame_index, frame in enumerate(trace):
            state = frame["snapshot"]
            trajectory.append(state.robot_state[0, :2].copy())
            qp_viz = None
            if hasattr(module, "build_qp_context"):
                context = module.build_qp_context(
                    jnp.asarray(frame["history"].reshape(1, -1)), hparams)
                qp_viz = dict(topk_ids=np.asarray(context["qp_topk_ids"][0]),
                              lookahead_distance=getattr(hparams, "lookahead_distance", 0.))
            status = frame["status"]
            if frame_index == len(trace)-1 and status == "running":
                status = "timeout"
            frames.append(render_jax_social_nav_frame(
                env_cfg, state, status=status, robot_traj=trajectory, qp_viz=qp_viz))
        status = ("collision" if record["collision"] else "success" if record["success"]
                  else "timeout")
        filename = f"episode_{index:04d}_seed_{record['seed']}_{status}.mp4"
        save_video(video_dir/filename, frames, fps=int(round(1./env_cfg.dt)))
    print(f"Videos: {video_dir}")
