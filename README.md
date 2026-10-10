# diff_rl_safe

JAX PPO for dynamic obstacle avoidance with six policies: BarrierNet, DiffCVaR,
and PPO, each using an MLP or GNN encoder. BarrierNet-GNN is the default.

This repository includes the JAX crowd environment, differentiable QP solver,
training, W&B logging, checkpoint evaluation, and ordinary rollout videos.
MuJoCo integration is a later milestone. Historical data and trained models are
not included.

## Setup

The CPU dependency set is tested with Python 3.10.16. With `uv`:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
source .venv/bin/activate
```

Alternatively, use `conda env create -f environment.yml`, activate
`diff_rl_safe`, and install `pytest` if running tests.

`requirements.txt` installs CPU JAX. For CUDA, install the matching accelerator
dependencies for JAX 0.4.35 using the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then launch with `device=cuda`. CUDA execution has not been validated here.
`device=auto` uses an available JAX backend. The launcher enables float64 for the
QP solver before importing JAX.

## Train and compare methods

Log an online run to your W&B account:

```bash
wandb login
WANDB_PROJECT=diff_rl_safe RUN_NAME=barriernet_gnn bash scripts/run_ppo.sh
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
RUN_NAME=diff_cvar_gnn bash scripts/run_ppo.sh model=diff_cvar_gnn
RUN_NAME=barriernet_si bash scripts/run_ppo.sh robot=single_integrator
```

Defaults retain the source settings: unicycle, 20 moving obstacles, 0.1-second
steps, three observation frames, top-five QP obstacles, and residual reward
weight 0.05. Training runs for 20 million environment steps by default.
For a short local run:

```bash
WANDB_MODE=disabled RUN_NAME=quick_check bash scripts/run_ppo.sh \
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
python scripts/eval.py --save-dir outputs/crowd_dyn_var_num_env/runs/<run> \
  --checkpoint outputs/crowd_dyn_var_num_env/runs/<run>/ckpt_00000512.pkl \
  --device cpu --seeds 100,200 --episodes 1 --visualize
```

Omit `--checkpoint` to evaluate all saved checkpoints, or `--visualize` to skip
videos. Episode counts are **per base seed**. The default is 50 episodes per
seed across ten seeds, so choose a small count when rendering videos.

`eval_results.json` is saved with or without videos. Videos capture the same
rollouts used to calculate those results: blue robot, yellow goal, gray
obstacles, red selected QP obstacles, and the robot trajectory. No advanced
diagnostic dashboard is included. The aggregate summarizes checkpoints, not
independent training seeds; compare named checkpoint results using the same
environment settings and test seeds.

Report success, collision, timeout, return, episode length, and minimum
clearance. `traj_time` and `min_dist` preserve the source's successful-episode
statistics; `mean_min_clearance` includes all episodes.
Missing clearance (for episodes with no active obstacles) and non-finite local
log values are represented as JSON `null`.

## Validation

```bash
WANDB_MODE=disabled python -m pytest tests -q
```

Tests cover all six policies and both robot dynamics, masked and changing QP
obstacles, gradients through learned α, checkpoint loading, two PPO updates
per policy, W&B payloads, and matching numerical/video evaluation. Smoke tests
use small networks and short episodes; they do not measure final performance.
See `VALIDATION.md` for source parity and recorded verification results.

## Implementation

Six explicit policy modules share MLP/GNN encoders, observation geometry,
safety-head helpers, and QP assembly. Deterministic CBF and CVaR constraints
remain separate. `model/` and `solver/` do not import the environment or trainer.

BarrierNet learns per-obstacle α and uses geometric safety distances with
observed obstacle velocities. DiffCVaR learns risk allocation and safety
distances using the source three-component motion predictor. The cleanup
preserves the original numerical operations, random-key allocation, parameter
trees, policy sampling, action transformations, and reward calculation.

The existing slack and Gaussian/tanh action path remains unchanged; a QP mean
is transformed and sampled before execution. Checkpoints contain parameters
and step, not optimizer/RNG state for exact training resumption. DiffCVaR
retains `gmm_weights` for compatibility; the current constraint combines
component risks using an unweighted smooth maximum.

Source attribution and excluded components are recorded in `PROVENANCE.md`.
