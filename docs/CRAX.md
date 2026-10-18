# CRAX Point crowd backend

This optional backend uses CRAX's Point scene builder and MJX pipeline. Our six
explicit policies, QPs, PPO update, social-force/GMM crowd, rewards and fixed-ID
observations remain in this repository. CRAX training algorithms are not used.
The original `env=crowd_dyn_var_num_env` baseline remains available.

## Isolated installation

```bash
micromamba create -n diffrl_crax -c conda-forge python=3.11 pip mesalib=21.2.5 -y
micromamba run -n diffrl_crax python -m pip install -r requirements-crax.txt
micromamba run -n diffrl_crax python -m pip check
```

Keep these requirements separate from `diffrl`, which pins an older JAX/NumPy
stack for the pure JAX baseline. CRAX uses MuJoCo/MJX 3.11.0 and JAX 0.6.0. CRAX is installed from the pinned commit in its environment,
without vendoring it. A resolved pip lock and a Linux x86_64 Conda explicit lock are provided alongside the requirements.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 RUN_NAME=crax_point_bnet WANDB_MODE=online \
  micromamba run -n diffrl_crax bash scripts/run_ppo.sh \
  env=crax_point_crowd robot=point model=barriernet_gnn device=cuda
```

Defaults are 128 environments × 32 control steps, 4,096 transitions, 4 PPO epochs,
20 million aggregate environment steps, top-five QP obstacles, three history
frames and residual reward weight 0.05. `model=barriernet_mlp`, `diff_cvar_gnn`,
`diff_cvar_mlp`, `ppo_gnn` and `ppo_mlp` select the other methods with their own
configured histories. W&B supports online/offline/disabled; the default project
is `diff_rl_safe`. Set `WANDB_ENTITY` explicitly when needed.

Control steps are 0.1 s, with 50 × 0.002 s MJX substeps. The speed/turn controller
is recomputed every physics substep. Update order matches the original JAX task:
the robot advances first while obstacles remain at their previous positions;
crowd goals, velocities and positions then update using the actual robot
endpoint. Obstacle positions are not interpolated through robot substeps.
Clearance and physical contacts include every robot substep and the final crowd
update. A final forward pass refreshes contacts without advancing time; forces
from new crowd contacts affect the next physics step. Physical cylinders
stay cylinders. Inactive obstacles are moved outside the scene; their graph
slots remain masked. Reset replaces arrays, never the XML/model.

`collision` is the 0.3 m planning envelope crossing the obstacle surface.
`physical_contact` is a native Point geom/crowd contact with nonpositive contact
distance. They have different geometry and are logged separately. CRAX's default
goal respawning, lidar, rewards and circular gremlin trajectories are not used.
Policy sampling and the post-QP action mapping are preserved, so this experiment
measures avoidance behavior without claiming strict QP constraint enforcement.

Timeouts bootstrap from the final observation and end the advantage trace.
Success/collision terminate without bootstrap. Completed environments receive
fresh reset keys and fresh repeated initial histories.

## Evaluate and render the same episodes

```bash
MUJOCO_GL=osmesa micromamba run -n diffrl_crax python scripts/eval.py \
  --save-dir <run-directory> --checkpoint <checkpoint-path> \
  --seeds 100,200 --episodes 5 --device cuda --visualize
```

Evaluation is compiled and deterministic. JSON metrics and MP4s come from one
rollout. Videos reconstruct the recorded MJX qpos/qvel/mocap arrays in the native
renderer; they do not run a second simulation. Videos include the goal, robot,
planning envelope, trajectory and selected QP obstacles. NPZ traces and per-episode
JSON make this relationship inspectable. Renderer settings apply to this command.

## Validate and benchmark

```bash
micromamba run -n diffrl_crax bash -c \
  'WANDB_MODE=disabled MUJOCO_GL=osmesa LD_PRELOAD="$CONDA_PREFIX/lib/libOSMesa.so" python -m pytest tests -q'
CUDA_VISIBLE_DEVICES=0 micromamba run -n diffrl_crax python scripts/validate_crax_encounters.py
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled micromamba run -n diffrl_crax \
  python scripts/benchmark_crax.py --updates 3 --name crax_benchmark
```

Benchmark reports synchronize full PPO updates, separate compilation from warm
execution for CRAX, and record device memory and an estimated 20M-step duration.
The estimate excludes evaluation/video overhead and depends on hardware/load.
Point is the first supported physical robot; other robots need a controller and
matching CBF dynamics before they can be selected.

Validation measurements and the first checkpoint are described in
[CRAX_VALIDATION.md](CRAX_VALIDATION.md). Its original timing, encounter and
checkpoint measurements used the previous interpolated crowd update and are
historical results, rather than measurements of the robot-first update.
