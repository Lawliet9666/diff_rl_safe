# diff_rl_safe

Social navigation with JAX environments, CBF-QP controllers, and PPO policies
using BarrierNet, DiffCVaR, or PPO with MLP/GNN encoders.

## Getting Started

For macOS (CPU), using Micromamba:

```bash
micromamba create -f environment.yml
micromamba activate diffrl
```

## Quick Start

Run a CBF-QP controller without training:

```bash
python scripts/run_controller.py robot=unicycle --visualize
python scripts/run_controller.py robot=quadrotor --visualize
```

Outputs are saved under `outputs/controllers/<robot>_<scenario>_seed<seed>/<timestamp>/`.
Omit `--visualize` to skip videos.

Train the default BarrierNet-GNN policy:

```bash
WANDB_MODE=disabled RUN_NAME=barriernet_gnn bash scripts/run_ppo.sh
```

Select another policy or robot:

```bash
WANDB_MODE=disabled RUN_NAME=diff_cvar_gnn bash scripts/run_ppo.sh model=diff_cvar_gnn
WANDB_MODE=disabled RUN_NAME=ppo_mlp_si bash scripts/run_ppo.sh model=ppo_mlp robot=single_integrator
```

PPO supports `unicycle` and `single_integrator`; the quadrotor currently uses the
controller entry point. Training outputs are saved under
`outputs/<scenario>/runs/<run>/`, including `config.yaml`, metrics, and checkpoints.

## Common Overrides

Pass Hydra overrides directly to either launcher where applicable:

| Option | Default | Override example |
| --- | --- | --- |
| Policy | `barriernet_gnn` | `model=diff_cvar_mlp` |
| Device | `auto` | `device=cpu` or `device=cuda` |
| Training environments | `128` | `trainer.num_envs=64` |
| Training timesteps | `20000000` | `trainer.total_timesteps=5000000` |
| Number of humans | `20` | `env.humans.num_humans=15` |
| Seed | `0` | `seed=1` |

Policies: `barriernet_gnn`, `barriernet_mlp`, `diff_cvar_gnn`, `diff_cvar_mlp`,
`ppo_gnn`, and `ppo_mlp`. See `config/` for all parameters.

For the optional CRAX/MJX backend, install its physics dependencies first, then run:

```bash
WANDB_MODE=disabled RUN_NAME=crax_barriernet bash scripts/run_ppo.sh env=crax_social_nav robot=unicycle
```

## W&B Logging

```bash
wandb login
WANDB_PROJECT=diff_rl_safe WANDB_ENTITY="<user_or_team>" \
  RUN_NAME=online_run bash scripts/run_ppo.sh
```

Use `WANDB_MODE=disabled` to disable W&B logging.

## Evaluation

Re-evaluate a saved controller run:

```bash
RUN_DIR="outputs/controllers/<robot>_<scenario>_seed<seed>/<timestamp>"
python scripts/eval.py --save-dir "$RUN_DIR" --seeds 100,200 --episodes 1 --visualize
```

Evaluate an RL checkpoint:

```bash
RUN_DIR="outputs/crowd_dyn_var_num_env/runs/<run>"
python scripts/eval.py --save-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/ckpt_<step>.pkl" --seeds 100,200 --episodes 1 --visualize
```

`--episodes` is the count per base seed. Omit `--checkpoint` to evaluate all RL
checkpoints, or `--visualize` to skip videos. Results are saved to
`<run>/eval_results.json`; videos go to `<run>/visualize_cbf_qp/` for controllers
and `<run>/visualize_ckpt_<step>/` for RL.

## Project Structure

```text
config/           Hydra environment, robot, controller, policy, and PPO settings
controller/       Deterministic CBF-QP controller and rollout execution
env/              JAX/CRAX environments and shared robot models
model/            BarrierNet, DiffCVaR, and PPO policies with MLP/GNN encoders
solver/jax_qpth/   Differentiable QP solver
trainer/          Shared PPO training and evaluation
utils.py          Shared device configuration and model checkpoint utilities
scripts/          Controller, training, and evaluation entry points
```
