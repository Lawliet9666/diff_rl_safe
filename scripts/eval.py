"""Evaluate RL checkpoints or a CBF-QP controller and optionally render its episodes."""

from pathlib import Path
from dataclasses import asdict, fields
import argparse
import json
import math
import os
import subprocess
import sys
import time

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


def _build_controller_config(config):
    from controller.cbf_qp import SolverConfig, UnicycleConfig, QuadrotorConfig, HumanoidConfig

    r, c, e = config.robot, config.controller, config.env
    if r.type not in {"unicycle", "quadrotor", "humanoid"}:
        raise ValueError("controller robot must be unicycle, quadrotor or humanoid")
    workspace = tuple(float(v) for v in [*e.workspace.x, *e.workspace.y])
    if (len(workspace) != 4 or not all(math.isfinite(v) for v in workspace)
            or workspace[0] >= workspace[1] or workspace[2] >= workspace[3]):
        raise ValueError("env.workspace requires finite increasing x/y bounds")
    def required_section(node, names, label):
        values = OmegaConf.to_container(node, resolve=True)
        missing = set(names)-values.keys()
        if missing:
            raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
        return values

    def robot_section(node, names, label):
        groups = ("nominal", "cbf", "qp")
        sections = required_section(node, groups, label)
        params = {key: value for key, value in sections.items() if key not in groups}
        blocks = [sections["nominal"], sections["qp"]]
        if r.type == "humanoid":
            cbf = required_section(OmegaConf.create(sections["cbf"]),
                                   ("distance", "parabolic"), f"{label}.cbf")
            blocks.extend([cbf["distance"], cbf["parabolic"]])
        else:
            blocks.append(sections["cbf"])
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"{label} parameter groups must be mappings")
            duplicate = params.keys() & block.keys()
            if duplicate:
                raise ValueError(f"{label} has duplicate parameters: {', '.join(sorted(duplicate))}")
            params.update(block)
        return required_section(OmegaConf.create(params), names, label)

    solver = SolverConfig(**required_section(c.solver, [f.name for f in fields(SolverConfig)],
                                            "controller.solver"))
    common = dict(radius=float(r.radius), margin=float(e.safety_margin),
                  omega_max=float(r.omega_max), solver=solver)
    if r.type == "unicycle":
        names = {f.name for f in fields(UnicycleConfig)}-common.keys()-{"v_max", "goal_tolerance"}
        params = robot_section(c.unicycle, names, "controller.unicycle")
        params["weights"] = tuple(params["weights"])
        return UnicycleConfig(goal_tolerance=float(e.robot_goal.tolerance),
                              v_max=float(r.v_max), **common, **params)
    elif r.type == "humanoid":
        common.pop("omega_max")  # Physical limits belong to the robot model.
        names = {f.name for f in fields(HumanoidConfig)}-common.keys()-{"dt", "goal_tolerance"}
        params = robot_section(c.humanoid, names, "controller.humanoid")
        for name in ("gains", "weights"):
            params[name] = tuple(params[name])
        return HumanoidConfig(dt=float(e.dt), goal_tolerance=float(e.robot_goal.tolerance),
                              **common, **params)
    else:
        physical = {name: float(r[name]) for name in
                    ["mass", "gravity", "thrust_rate_max", "thrust_min", "thrust_max",
                     "tilt_max", "altitude", "altitude_min", "altitude_max"]}
        names = {f.name for f in fields(QuadrotorConfig)}-common.keys()-physical.keys()
        params = robot_section(c.quadrotor, names, "controller.quadrotor")
        for name in ("gains", "weights", "acceleration_limits"):
            params[name] = tuple(params[name])
        return QuadrotorConfig(**common, **physical, **params)


def _prepare_controller(config):
    """Validate settings and create the shared parameters and action generator."""
    from controller import make_controller

    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ValueError("seed must be an integer")
    if config.env.get("backend", "jax") != "jax":
        raise ValueError("controller runner requires the original JAX crowd backend")
    if config.env.humans.use_gmm:
        raise ValueError("controller runner requires GMM noise disabled")
    if not math.isfinite(config.env.dt) or config.env.dt <= 0:
        raise ValueError("env.dt must be positive")
    if (not isinstance(config.env.max_steps, int) or isinstance(config.env.max_steps, bool)
            or config.env.max_steps <= 0):
        raise ValueError("env.max_steps must be a positive integer")
    if config.robot.type in {"quadrotor", "humanoid"}:
        max_dt = config.robot.max_integration_dt
        if not math.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("robot.max_integration_dt must be positive")
    if "simulation" in config:
        raise ValueError("simulation overrides are no longer supported; configure env.* instead")
    if config.env.humans.num_humans < 1:
        raise ValueError("env.humans.num_humans must be >=1")
    return _build_controller_config(config), make_controller(config)


def _controller_failure_diagnostics(state, nominal, obs, result):
    def values(x):
        return [float(v) if np.isfinite(v) else None for v in np.asarray(x).reshape(-1)]
    diagnostic = dict(solver="osqp", solver_status=result.solver_status,
        iterations=result.iterations, state=values(state), nominal=values(nominal),
        candidate=values(result.candidate), kkt_residual=values(result.kkt_residual),
        residual=values(result.residual), centers=values(obs.centers),
        velocities=values(obs.velocities), radii=values(obs.radii), mask=values(obs.mask))
    if result.solver_status == "invalid barrier domain":
        diagnostic["solver"] = "not_run"
        return "controller_failure", "parabolic barrier requires separation beyond the inflated radius", diagnostic
    if result.solver_status == "primal infeasible":
        return "infeasible", "OSQP reports incompatible hard constraints", diagnostic
    reason = ("OSQP solution failed independent residual checks"
              if result.solver_status == "solved" else "OSQP: " + result.solver_status)
    return "controller_failure", reason, diagnostic


def _run_controller_episode(config, *, controller_setup=None):
    """Run a fresh episode; reuse an optional batch (parameters, generator) pair."""
    import jax
    import jax.numpy as jnp
    from controller.cbf_qp import control_residual
    from env.jax_env.jax_social_nav import compute_distances
    from controller.env_adapter import NavigationEnv

    rcfg, generate_action = (_prepare_controller(config) if controller_setup is None
                             else controller_setup)
    robot, scenario, seed = str(config.robot.type), str(config.env.scenario), int(config.seed)
    env = NavigationEnv(config, state_tolerance=rcfg.solver.feasibility_tolerance)
    cfg = env.cfg
    navigation_state = env.reset(jax.random.PRNGKey(seed))
    state, crowd = navigation_state
    goal = env.observe(navigation_state).goal
    robot_model = env.robot

    def action_fn(observation):
        return generate_action(robot_model, observation.robot, observation.goal,
                               observation.obstacles, rcfg)

    action_fn(env.observe(navigation_state))[0].block_until_ready()
    history = {k: [] for k in ["time", "state", "centers", "status", "active_obstacles",
        "nominal", "action", "requested_action", "min_residual", "initial_safe", "kkt_residual", "solve_ms",
        "qp_valid", "fallback", "solver_status", "slack", "cbf_satisfied",
        "min_cbf_residual", "min_relaxed_cbf_residual", "min_input_residual"]}
    status, reason = "timeout", "environment step limit"
    worst_clearance, min_residual = float("inf"), float("inf")
    violations, failures, path_length = 0, 0, 0.
    infeasible_steps, fallback_steps = 0, 0
    previous_action = jnp.zeros(robot_model.control_dim, dtype=state.dtype)
    failure = None
    key = jax.random.PRNGKey(seed)

    def record_frame(label):
        nonlocal worst_clearance
        _, clearance = compute_distances(cfg, crowd)
        worst_clearance = min(worst_clearance, float(clearance[0]))
        history["time"].append(float(crowd.step_count[0])*cfg.dt)
        history["state"].append(np.asarray(state).copy())
        history["centers"].append(np.asarray(crowd.human_positions[0]).copy())
        history["status"].append(label)
        history["active_obstacles"].append(np.zeros(cfg.num_humans, dtype=bool))

    record_frame("running")
    for k in range(cfg.max_steps):
        observation = env.observe(navigation_state)
        obs = observation.obstacles
        start = time.perf_counter()
        action, diagnostics = action_fn(observation)
        action.block_until_ready()
        # Preserve the timing field; it now includes nominal action generation.
        solve_ms = (time.perf_counter()-start)*1000
        nominal, result = diagnostics["nominal"], diagnostics["qp"]
        violations += int(not bool(result.initial_safe))
        used_fallback = not bool(result.valid)
        if used_fallback:
            failures += 1
            qp_status, qp_reason, failure = _controller_failure_diagnostics(state, nominal, obs, result)
            failure["step"] = k
            if robot == "humanoid" and qp_status == "infeasible":
                infeasible_steps += 1
                status, reason = qp_status, qp_reason
                break  # An old acceleration is not a safe fallback.
            if qp_status != "infeasible" or config.controller.type == "cbf_qp_slack":
                status = "controller_failure" if config.controller.type == "cbf_qp_slack" else qp_status
                reason = qp_reason
                break
            infeasible_steps += 1
            action = previous_action
        if k == 0 and not bool(result.initial_safe):
            status, reason = "invalid_initial_state", "initial h/psi safety conditions are not satisfied"
            break
        requested = action
        previous = state
        key, motion_key = jax.random.split(key)
        navigation_state, _, done, metrics = env.step(navigation_state, requested, motion_key)
        state, crowd = navigation_state
        executed = metrics["applied_action"]
        previous_action = executed
        fallback_steps += int(used_fallback)
        applied_residual = np.asarray(control_residual(robot_model, previous, executed, obs, rcfg))
        residual = float(np.min(applied_residual))
        min_residual = min(min_residual, residual)
        slack = np.asarray(diagnostics["slack"])
        cbf_residual = applied_residual[:cfg.num_humans]
        cbf_satisfied = bool(np.all(cbf_residual >= -rcfg.solver.feasibility_tolerance))
        history["active_obstacles"][-1] = (np.abs(cbf_residual+slack)
            <= rcfg.solver.feasibility_tolerance) & (np.asarray(obs.mask) > .5)
        path_length += float(np.linalg.norm(np.asarray(state[:2])-np.asarray(previous[:2])))
        for name, value in dict(nominal=nominal, action=executed, requested_action=requested,
                min_residual=residual, initial_safe=bool(result.initial_safe),
                kkt_residual=result.kkt_residual,
                solve_ms=solve_ms, qp_valid=bool(result.valid), fallback=used_fallback,
                solver_status=result.solver_status, slack=slack, cbf_satisfied=cbf_satisfied,
                min_cbf_residual=float(np.min(cbf_residual)),
                min_relaxed_cbf_residual=float(np.min(cbf_residual+slack)),
                min_input_residual=float(np.min(applied_residual[-2*robot_model.control_dim:]))).items():
            history[name].append(np.asarray(value).copy())
        record_frame("running")
        if bool(metrics["collision"][0]):
            status, reason = "collision", "physical envelopes overlap at an environment step"
            break
        if bool(metrics["physical_state_limit"]):
            status = "state_limit"
            reason = ("humanoid body velocity limit exceeded during integration" if robot == "humanoid"
                      else "quadrotor altitude, tilt or thrust limit exceeded during integration")
            break
        if bool(done[0]):
            status = "success" if bool(metrics["success"][0]) else "timeout"
            reason = "goal reached under the original distance criterion" if status == "success" else "environment step limit"
            break
    history["status"][-1] = status
    width = robot_model.control_dim
    history = {k: np.asarray(v) for k, v in history.items()}
    for name in ["nominal", "action", "requested_action"]:
        history[name] = history[name].reshape((-1, width))
    history["kkt_residual"] = history["kkt_residual"].reshape((-1, 3))
    history["slack"] = history["slack"].reshape((-1, cfg.num_humans))
    finite = lambda value: float(value) if np.isfinite(value) else None
    dist, _ = compute_distances(cfg, crowd)
    integration_dt = env.integration_dt
    summary = dict(robot=robot, scenario=scenario, seed=seed, status=status, reason=reason,
        steps=len(history["action"]), min_clearance=finite(worst_clearance),
        min_constraint_residual=finite(min_residual), qp_failures=failures,
        qp_infeasible_steps=infeasible_steps, fallback_steps=fallback_steps,
        initial_condition_violation_steps=violations, failure_diagnostics=failure,
        goal_distance=float(dist[0]), path_length=path_length,
        mean_solve_ms=finite(np.mean(history["solve_ms"])) if len(history["solve_ms"]) else None,
        dt=cfg.dt, integration_dt=integration_dt, crowd_dt=cfg.dt,
        max_steps=cfg.max_steps, goal_tolerance=cfg.goal_tolerance, num_humans=cfg.num_humans,
        clearance_sampling="environment_step",
        config={**asdict(rcfg), "workspace": tuple(float(v) for v in
            [*config.env.workspace.x, *config.env.workspace.y])})
    summary.update(
        controller=str(config.controller.type),
        slack_weight=config.controller.get("slack_weight"),
        slack_steps=int(np.count_nonzero(np.any(
            history["slack"] > rcfg.solver.feasibility_tolerance, axis=1))),
        slack_sum=finite(np.sum(history["slack"])),
        max_slack=finite(np.max(history["slack"])) if history["slack"].size else 0.,
        cbf_violation_steps=int(np.count_nonzero(~history["cbf_satisfied"].astype(bool))),
        **{name: finite(np.min(history[name])) if history[name].size else None for name in
           ("min_cbf_residual", "min_relaxed_cbf_residual", "min_input_residual")})
    return (summary, history, np.asarray(goal), crowd), cfg, rcfg


def _render_controller_frame(cfg, history, index, crowd, robot, robot_cfg):
    """Only adapt state; all pixels/layout come from the existing RL renderer."""
    import jax.numpy as jnp
    from controller.env_adapter import project_quadrotor, project_humanoid

    from env.jax_env.jax_render import render_jax_social_nav_frame
    state = jnp.asarray(history["state"][index])
    if robot == "quadrotor":
        frame_state = project_quadrotor(crowd, state)
    elif robot == "humanoid":
        frame_state = project_humanoid(crowd, state)
    else:
        frame_state = crowd._replace(robot_state=state[None, :])
    frame_state = frame_state._replace(
        human_positions=jnp.asarray(history["centers"][index])[None, ...],
        step_count=jnp.array([round(float(history["time"][index])/cfg.dt)], jnp.int32))
    qp_viz = dict(topk_ids=np.flatnonzero(history["active_obstacles"][index]),
                  lookahead_distance=robot_cfg.lookahead if robot == "unicycle" else 0.)
    return render_jax_social_nav_frame(cfg, frame_state, status=str(history["status"][index]),
        robot_traj=history["state"][:index+1, :2], qp_viz=qp_viz, robot_type=robot)


def _save_controller_video(path, cfg, history, crowd, robot, robot_cfg):
    # The existing encoder accepts an indexable iterable. Render lazily to avoid
    # retaining hundreds of full-resolution frames in the controller-only run.
    class Frames:
        def __getitem__(self, index):
            if index >= len(history["state"]):
                raise IndexError(index)
            return _render_controller_frame(cfg, history, index, crowd, robot, robot_cfg)
    save_video(path, Frames(), fps=1./cfg.dt)


def _controller_evaluation_metrics(result):
    """Use eval metric names for one episode; retain controller failures separately."""
    status = result["status"]
    return dict(status=status,
        # Controller rollouts do not record a comparable RL episode return.
        mean_return=None, std_return=None,
        success_rate=float(status == "success"),
        collision_rate=float(status == "collision"),
        timeout_rate=float(status == "timeout"),
        failure_rate=float(status not in {"success", "collision", "timeout"}),
        mean_episode_length=float(result["steps"]),
        mean_min_clearance=result["min_clearance"],
        std_min_clearance=0.0 if result["min_clearance"] is not None else None,
        qp_failures=result["qp_failures"], mean_solve_ms=result["mean_solve_ms"],
        infeasible_rate=float(result["qp_infeasible_steps"] > 0), fallback_steps=result["fallback_steps"])


def evaluate_controller(config, output, *, visualize=False, seeds=None, episodes=1):
    """Shared controller evaluation entry for Hydra and saved-run evaluation."""
    from utils import configure_jax_platform_for_device, validate_jax_backend_for_device

    platform = configure_jax_platform_for_device(config.device)
    import jax
    jax.config.update("jax_enable_x64", True)
    if platform != "auto":
        jax.config.update("jax_platforms", platform)
    validate_jax_backend_for_device(jax, config.device)

    from trainer.evaluation import eval_seed_values, safe_mean_std, paper_metric_summary

    controller_setup = _prepare_controller(config)
    seed_values = eval_seed_values(episodes, [int(config.seed)] if seeds is None else seeds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output/"config.yaml", resolve=True)
    controller = str(config.controller.type)
    results = []
    episode_path = output/"episodes.jsonl"
    episode_path.write_text("")
    for index, seed in enumerate(seed_values):
        episode_config = OmegaConf.merge(config, {"seed": int(seed)})
        (result, history, goal, crowd), cfg, rcfg = _run_controller_episode(
            episode_config, controller_setup=controller_setup)
        results.append(result)
        with episode_path.open("a") as episode_log:
            episode_log.write(json.dumps(result, allow_nan=False)+"\n")
        if (index+1) % 10 == 0:
            print(f"{controller}: completed {index+1}/{len(seed_values)} episodes", flush=True)
        if visualize:
            video_dir = output/f"visualize_{controller}"
            video_dir.mkdir(parents=True, exist_ok=True)
            filename = f"episode_{index:04d}_seed_{int(seed)}_{result['status']}.mp4"
            _save_controller_video(video_dir/filename, cfg, history, crowd, result["robot"], rcfg)
            print(f"Video: {video_dir/filename}")

    episode_metrics = [_controller_evaluation_metrics(result) for result in results]
    metrics = dict(episode_metrics[0])
    if len(results) > 1:
        statuses = {result["status"] for result in results}
        metrics["status"] = results[0]["status"] if len(statuses) == 1 else "mixed"
        for key in ("success_rate", "collision_rate", "timeout_rate", "failure_rate", "infeasible_rate", "mean_episode_length"):
            metrics[key] = float(np.mean([m[key] for m in episode_metrics]))
        metrics["mean_min_clearance"], metrics["std_min_clearance"] = safe_mean_std(
            [np.nan if r["min_clearance"] is None else r["min_clearance"] for r in results])
        metrics["qp_failures"] = sum(r["qp_failures"] for r in results)
        metrics["fallback_steps"] = sum(r["fallback_steps"] for r in results)
        timed = [r for r in results if r["mean_solve_ms"] is not None]
        steps = sum(r["steps"] for r in timed)
        metrics["mean_solve_ms"] = (sum(r["mean_solve_ms"]*r["steps"] for r in timed)/steps
                                    if steps else None)
    metrics.update(paper_metric_summary(
        [r["status"] == "success" for r in results],
        np.asarray([r["steps"] for r in results]),
        np.asarray([np.nan if r["min_clearance"] is None else r["min_clearance"] for r in results]),
        float(config.env.dt)))
    steps = sum(r["steps"] for r in results)
    metrics.update(
        slack_steps=sum(r["slack_steps"] for r in results),
        cbf_violation_steps=sum(r["cbf_violation_steps"] for r in results),
        mean_slack=sum(r["slack_sum"] for r in results)/steps if steps else None,
        max_slack=max(r["max_slack"] for r in results))
    for name in ("min_cbf_residual", "min_relaxed_cbf_residual", "min_input_residual"):
        values = [r[name] for r in results if r[name] is not None]
        metrics[name] = min(values) if values else None
    metrics["slack_step_rate"] = metrics["slack_steps"]/steps if steps else None
    metrics["cbf_violation_rate"] = metrics["cbf_violation_steps"]/steps if steps else None
    aggregate = {"num_controllers": 1}
    if len(results) > 1:
        aggregate["num_episodes"] = len(results)
    for key in ("mean_return", "success_rate", "collision_rate", "timeout_rate", "failure_rate", "infeasible_rate", "mean_min_clearance"):
        aggregate[f"mean_{key}"] = metrics[key]
    results_path = output/"eval_results.json"
    results_path.write_text(json.dumps({controller: metrics, "aggregate": aggregate},
        indent=2, allow_nan=False)+"\n")
    print(f"{controller}: {json.dumps(metrics, allow_nan=False)}")
    print(f"Results: {results_path}")
    if any(r["status"] in {"controller_failure", "infeasible", "invalid_initial_state", "collision", "state_limit"}
           for r in results):
        raise SystemExit(1)
    return results[0] if len(results) == 1 else results


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
        from utils import configure_jax_platform_for_device
        configure_jax_platform_for_device(self.config.device)

    def evaluate_one_ckpt(self, checkpoint):
        from env.jax_env.jax_social_nav import config_from_omegaconf
        from model import get_model_module
        from utils import load_jax_checkpoint
        from env import make_env
        from trainer.evaluation import evaluate_env

        checkpoint = Path(checkpoint)
        params = load_jax_checkpoint(checkpoint)["params"]
        module = get_model_module(self.config.model)
        hparams = module.hparams_from_config(self.config.model, self.config.model.critic, act_dim=2)
        env_cfg = config_from_omegaconf(self.config)
        env = make_env(self.config)
        video_dir = self.save_dir/f"visualize_{checkpoint.stem}" if self.visualize else None
        return evaluate_env(env, params, hparams, env_cfg, module,
                            episodes=self.episodes, seeds=self.seeds, video_dir=video_dir)

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
