# diff_rl_safe

JAX PPO for dynamic obstacle avoidance with six policies: BarrierNet, DiffCVaR,
and PPO, each using an MLP or GNN encoder. BarrierNet-GNN is the default.

This repository includes the JAX crowd environment, differentiable QP solver,
training, W&B logging, checkpoint evaluation, and ordinary rollout videos.
CRAX/MJX provides the optional compiled Point physics backend. Both environments
support all six policies. Historical data and trained models are not included.

## Setup

Use Micromamba for setup. The `diffrl` environment runs the original JAX crowd
backend. Compiled CRAX/MJX physics uses the separate `diffrl_crax` environment; see
[the CRAX setup instructions](docs/CRAX.md#isolated-installation).

Create `diffrl`. For controller-only checks, install `requirements-controller.txt`
as shown in [the controller guide](docs/CONTROLLERS.md#run-locally). For the full
RL dependency set, use:

```bash
micromamba create -f environment.yml
micromamba run -n diffrl python -m pip install -r requirements-dev.txt
micromamba run -n diffrl python -m pip check
```

For NVIDIA GPU policy/QP/PPO computation, add the matching optional CUDA
packages to the same environment:

```bash
micromamba run -n diffrl python -m pip install -r requirements-gpu.txt
micromamba run -n diffrl python -m pip check
```

This preserves JAX/JAXlib 0.4.35 and NumPy 1.26.4. The optional file pins the
JAX CUDA plugin and CUDA 12.6 libraries. Choose a GPU through a command-local
`CUDA_VISIBLE_DEVICES` and use `device=cuda` for training. Point physics and
headless rendering use the separate CRAX installation described below.

The full RL CPU dependency set was previously tested with Python 3.10.16.
The current environment specification uses Python 3.11; the controller-only
dependency set is tested with Python 3.11.17.

`requirements.txt` installs CPU JAX. For CUDA, install the matching accelerator
dependencies for JAX 0.4.35 using the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then launch with `device=cuda`. CUDA execution has not been validated here.
`device=auto` uses an available JAX backend. The launcher enables float64 for the
QP solver before importing JAX.

## Controller-only CBF-QP checks

Unicycle and attitude-aware quadrotor filters can run in the existing crowd
without RL training or checkpoints. See [controller setup, equations and
validation](docs/CONTROLLERS.md).

```bash
micromamba run -n diffrl python scripts/run_controller.py \
  robot=quadrotor --visualize
```

Optional diagnostics select `simulation.scenario` (`empty`, `static`, `crossing`)
and require explicit `simulation` settings; normal runs use the selected environment
and its original reset. Omit `--visualize` to skip video recording.

## Train and compare methods

Log an online run to your W&B account:

```bash
micromamba run -n diffrl wandb login
WANDB_PROJECT=diff_rl_safe RUN_NAME=barriernet_gnn \
  micromamba run -n diffrl bash scripts/run_ppo.sh
```

Set `WANDB_ENTITY` to select a user or team. `WANDB_MODE=offline` saves logs for
later synchronization; `WANDB_MODE=disabled` runs without W&B synchronization.
No account, API key, or team is embedded in this repository.

| Policy | Training override |
| --- | --- |
| BarrierNet-GNN | `model=barriernet_gnn` (default) |
| BarrierNet-MLP | `model=barriernet_mlp` |
| DiffCVaR-GNN | `model=diff_cvar_gnn` |
| DiffCVaR-MLP | `model=diff_cvar_mlp` |
| PPO-GNN | `model=ppo_gnn` |
| PPO-MLP | `model=ppo_mlp` |

```bash
RUN_NAME=diff_cvar_gnn micromamba run -n diffrl bash scripts/run_ppo.sh model=diff_cvar_gnn
RUN_NAME=barriernet_si micromamba run -n diffrl bash scripts/run_ppo.sh robot=single_integrator
```

Defaults retain the source settings: unicycle, 20 moving obstacles, 0.1-second
steps, three observation frames, top-five QP obstacles, and residual reward
weight 0.05. Training runs for 20 million environment steps by default.

New training uses `trainer.timeout_bootstrap=true` in both environments and all
six methods. Episodes still end and reset at 400 steps. The value target uses
the final observation before reset; GAE never carries across that reset.
Success and collision remain terminal without bootstrap. Set
`trainer.timeout_bootstrap=false` to restore the legacy treatment of the
400-step limit as terminal. This changes learning targets, not rewards, actions
or the episode limit, and is recorded in the run's resolved config and W&B.
For old resolved configs lacking this field, JAX retains terminal timeouts and
CRAX retains timeout bootstrap. Choose the same explicit value for comparisons.

```bash
RUN_NAME=jax_bootstrap micromamba run -n diffrl bash scripts/run_ppo.sh trainer.timeout_bootstrap=true
RUN_NAME=jax_terminal micromamba run -n diffrl bash scripts/run_ppo.sh trainer.timeout_bootstrap=false
```

For a short local run:

```bash
WANDB_MODE=disabled RUN_NAME=quick_check micromamba run -n diffrl bash scripts/run_ppo.sh \
  device=cpu trainer.num_envs=8 trainer.timesteps_per_batch=256 \
  trainer.total_timesteps=512 trainer.num_minibatches=4 \
  trainer.n_updates_per_iteration=1 trainer.finite_check_interval=1 \
  trainer.eval_interval=1 trainer.eval_episodes=1
```

Each run writes `config.yaml`, `metrics.jsonl`, checkpoints, and a checkpoint
manifest below `outputs/crowd_dyn_var_num_env/runs/`. W&B retains training losses,
evaluation success/collision/timeout rates, clearance, and QP metrics. Safety
policies also log learned α or β and safety distances.

## Evaluate and watch obstacle avoidance

Replace the paths with the run directory and checkpoint you want to test:

```bash
micromamba run -n diffrl python scripts/eval.py --save-dir outputs/crowd_dyn_var_num_env/runs/<run> \
  --checkpoint outputs/crowd_dyn_var_num_env/runs/<run>/ckpt_00000512.pkl \
  --device cpu --seeds 100,200 --episodes 1 --visualize
```

For a CRAX Point checkpoint, use `diffrl_crax` with the same flags; see
[Point evaluation and rendering](docs/CRAX.md#evaluate-and-render-the-same-episodes).

Omit `--checkpoint` to evaluate all saved checkpoints, or `--visualize` to skip
videos. Episode counts are **per base seed**. The default is 50 episodes per
seed across ten seeds, so choose a small count when rendering videos.

`eval_results.json` is saved with or without videos. Videos capture the same
rollouts used to calculate those results: blue robot, visible goal, gray
obstacles, red selected QP obstacles, and the robot trajectory. No advanced
diagnostic dashboard is included. The aggregate summarizes checkpoints, not
independent training seeds; compare named checkpoint results using the same
environment settings and test seeds.

Both evaluators classify each episode once, with collision taking priority over
success and success over timeout. Collision uses the planning-envelope geometry;
CRAX physical contacts remain separate. This reporting rule does not change the
reward calculation or training outcome counters. Re-evaluate old checkpoints
with the same evaluator version for comparisons; saved historical results are
not rewritten.

Report success, collision, timeout, return, episode length, and minimum
clearance. `traj_time` and `min_dist` preserve the source's successful-episode
statistics; `mean_min_clearance` includes all episodes.
Missing clearance (for episodes with no active obstacles) and non-finite local
log values are represented as JSON `null`.

## Validation

```bash
WANDB_MODE=disabled micromamba run -n diffrl python -m pytest tests -q
```

CRAX Point tests are skipped when the optional CRAX dependency is absent.

Tests cover all six policies and both robot dynamics, masked and changing QP
obstacles, gradients through learned α, checkpoint loading, two PPO updates
per policy, W&B payloads, and matching numerical/video evaluation. Smoke tests
use small networks and short episodes; they do not measure final performance.
See `VALIDATION.md` for source parity and recorded verification results.

## Implementation

Six explicit policy modules share MLP/GNN encoders, observation geometry,
safety-head helpers, and QP assembly. Deterministic CBF and CVaR constraints
remain separate. `model/` and `solver/` do not import the environment or trainer.

BarrierNet-MLP and BarrierNet-GNN learn per-obstacle α and safety distances,
using observed obstacle velocities in the deterministic CBF. Their distance is
`r_robot + r_obstacle + margin_base + margin_extra_max * sigmoid(distance_logit)`.
With the default radii this ranges from 0.75 to 1.875 m, matching DiffCVaR's
distance parameterization. GNN predicts from each obstacle latent; MLP predicts
one value per persistent obstacle slot. Both heads receive gradients through
the QP. This is a learned-distance extension of the inherited BarrierNet.

Use `model.actor.learn_r_safe=false` to reproduce the fixed-distance variant.
Old resolved configs without this setting retain fixed distances and load
without an added parameter head; existing checkpoints do not gain learned
distances retroactively. New training defaults to the learned-distance variant.
The existing α, nominal-control and critic initialization keys are preserved.
Training already records `diagnostics/r_safe_mean` in local metrics and W&B.

DiffCVaR learns risk allocation and safety distances using the source
three-component motion predictor. BarrierNet retains a deterministic CBF and
does not gain DiffCVaR's β allocation or CVaR uncertainty terms. The original
cleanup preserved numerical operations and parameter trees; the subsequent
BarrierNet extension adds only the optional distance head. Policy sampling,
action transformations and reward calculation are unchanged.

The existing slack and Gaussian/tanh action path remains unchanged; a QP mean
is transformed and sampled before execution, so this experiment does not claim
strict constraint enforcement. Checkpoints contain parameters and step, not
optimizer/RNG state for exact training resumption. DiffCVaR
retains `gmm_weights` for compatibility; the current constraint combines
component risks using an unweighted smooth maximum.

Source attribution and excluded components are recorded in `PROVENANCE.md`.

## Optional CRAX/MJX Point crowd task

Use the isolated `diffrl_crax` environment for compiled, batched MuJoCo physics:

```bash
RUN_NAME=crax_point_bnet WANDB_MODE=online micromamba run -n diffrl_crax \
  bash scripts/run_ppo.sh env=crax_social_nav robot=point model=barriernet_gnn device=cuda
```

All six methods are selectable in both the original JAX baseline and CRAX Point.
`robot=point` requires `env=crax_social_nav`. Installation, evaluation/video
commands and synchronized benchmarking are documented in [docs/CRAX.md](docs/CRAX.md).
