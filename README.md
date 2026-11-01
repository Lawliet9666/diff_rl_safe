# diff_rl_safe
![Recorded unicycle navigation](docs/demo.gif)

JAX social navigation with CBF-QP controllers and PPO, CBF-RL, and BarrierNet policies.

## Prerequisites
- Micromamba for the local `diffrl` environment.
- CUDA-enabled JAX for GPU execution; Slurm and a Conda environment named `diffrl` on Matilda.
- W&B authentication (`wandb login`), or set `WANDB_MODE=disabled`.

## Getting Started
```bash
micromamba create -f environment.yml
micromamba activate diffrl
```

## Quick Start
```bash
# CBF controller, no training
bash scripts/run_local.sh cbf unicycle --visualize

# Preview CBF-RL across five robots
bash scripts/run_local.sh cbf_rl all --device cuda --dry-run

# Train PPO and PPO+filter sequentially on one GPU
bash scripts/run_ppo.sh both all --device cuda

# Preview four independent Matilda jobs
bash scripts/run_matilda.sh all --dry-run

# Submit the four jobs on Matilda
bash scripts/run_matilda.sh all
```

| Method | Script / selector |
| --- | --- |
| PPO / PPO + safety filter | `run_ppo.sh ppo`, `ppo_filter`, or `both`; local selector `ppo_both` runs both |
| CBF-RL with runtime filtering | `run_cbf_rl.sh` / `cbf_rl` |
| Differentiable CBF-QP | `run_barriernet.sh` / `barriernet` |
| Differentiable CBF-QP + simulation gradients | `run_barriernet_sim.sh` / `barriernet_sim` |
| CBF controller | `run_cbf.sh` / `cbf` |

All five robot models are supported: `single_integrator`, `unicycle`,
`double_integrator`, `humanoid`, and `quadrotor`. Methods and robots execute
sequentially within each process. Matilda `all` submits four independent jobs:
PPO+PPO-filter, CBF-RL, BarrierNet, and BarrierNet+simulation.

| Launcher | CPU threads | Environments | PPO batch | Steps per run |
| --- | ---: | ---: | ---: | ---: |
| Local CPU | 4 | 4 | 1024 | 15M |
| Local CUDA | 8 | 16 | 4096 | 15M |
| Matilda, per job | 8 | 32 | 4096 | 15M |

Each Matilda job requests one GPU and 64 GB RAM. W&B is online by default;
`RUN_NAME`, `WANDB_MODE`, `WANDB_ENTITY`, and `WANDB_PROJECT` customize logging.
GPU visibility is inherited from Slurm. Local `--gpu ID` selects one visible GPU.
`CPU_THREADS` sets thread limits, not CPU affinity. The default crowd has 30 humans.

Pass Hydra overrides after launcher options:
```bash
bash scripts/run_local.sh cbf_rl unicycle --device cpu model=ppo_mlp seed=1
bash scripts/run_matilda.sh cbf_rl all seed=1 trainer.total_timesteps=5000000
```
Robot/CBF parameters remain in `config/`. Additional policy modules, including
DiffCVaR, can be selected with `model=`; the optional CRAX backend uses
`env=crax_social_nav` with a supported robot. Training outputs go to
`outputs/<scenario>/runs/<run>/`; controllers use `outputs/controllers/`.

## Evaluation
```bash
RUN_DIR="outputs/crowd_dyn_var_num_env/runs/<run>"
python scripts/eval.py --save-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/ckpt_<step>.pkl" --seeds 100,200 --episodes 50 --visualize
```
`--episodes` is the count per base seed. Omit `--checkpoint` to evaluate all RL
checkpoints. The same evaluator accepts controller run directories. Results are
saved to `<run>/eval_results.json`, with videos under `<run>/visualize_*/`.

## Project Structure
```text
config/           Hydra environment, robot, controller, policy, and PPO settings
controller/       CBF-QP controllers and fixed execution filters
env/              JAX/CRAX environments and shared robot models
model/            PPO, BarrierNet, and DiffCVaR policies
solver/jax_qpth/   Differentiable QP solver
trainer/          Shared PPO training and evaluation
scripts/          run_local.sh, run_matilda.sh, method profiles, and Python entry points
utils.py          Device configuration and checkpoint utilities
```
