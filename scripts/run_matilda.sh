#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_matilda.sh GROUP [ROBOT] [Hydra overrides]

GROUP: all, ppo_both, ppo, ppo_filter, cbf_rl, barriernet, barriernet_sim.
ROBOT defaults to all five models; a single robot may also be selected.
all submits four independent jobs: PPO+PPO-filter, CBF-RL, BarrierNet,
and BarrierNet+simulation. Methods/robots run sequentially within each job.

Each job: 1 GPU, 8 CPUs, 64 GB RAM, normal QoS, three-day time limit.
All RL runs: GNN, 32 environments, batch 4096, 15M steps, 30 humans, W&B online.
Extra Hydra overrides take precedence. Authenticate W&B before submission.
HELP
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
selection="$1"
shift
target=all
if [[ $# -gt 0 && "$1" != -* && "$1" != *=* ]]; then target="$1"; shift; fi
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpu|--gpu=*|--device|--device=*) printf 'Slurm selects the GPU; remove %s.\n' "$1" >&2; exit 2 ;;
    --) shift; overrides+=("$@"); break ;;
    -*) printf 'Unknown launcher option: %s\n' "$1" >&2; exit 2 ;;
    *) overrides+=("$1"); shift ;;
  esac
done
# Preserve empty argument lists on the macOS Bash 3.2 runtime.
set -- ${overrides[@]+"${overrides[@]}"}
case "$selection" in
  all) groups=(ppo_both cbf_rl barriernet barriernet_sim) ;;
  ppo_both|ppo|ppo_filter|cbf_rl|barriernet|barriernet_sim) groups=("$selection") ;;
  *) printf 'Unknown Matilda group: %s\n' "$selection" >&2; exit 2 ;;
esac
case "$target" in
  single_integrator|unicycle|double_integrator|humanoid|quadrotor|all) ;;
  *) printf 'Unknown robot: %s\n' "$target" >&2; exit 2 ;;
esac
for argument in "$@"; do
  case "$argument" in
    robot=*|+robot=*|++robot=*|device=*|+device=*|++device=*|run_name=*|+run_name=*|++run_name=*)
      printf 'Launcher controls robot/device/run names; remove %s (use RUN_NAME for a prefix).\n' "$argument" >&2
      exit 2 ;;
  esac
done
root="$(cd "$(dirname "$0")/.." && pwd)"

job_script() {
  cat <<'SLURM'
#!/bin/bash -l
set -eo pipefail
module load miniconda3/current
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate diffrl
hash -r

python - <<'CHECK_PYTHON'
import os
import sys
expected = os.path.realpath(os.environ["CONDA_PREFIX"])
if os.path.realpath(sys.prefix) != expected:
    raise RuntimeError(f"Wrong Python: {sys.executable}; expected {expected}")
CHECK_PYTHON

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-diff_rl_safe}"
export RUN_NAME="${RUN_NAME:-matilda}_${SLURM_JOB_ID}"
export CPU_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/scratch/users/${USER}/tmp/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
SLURM
  printf 'exec '
  printf '%q ' bash "$root/scripts/run_local.sh" "$group" "$target" --device cuda \
    trainer.num_envs=32 trainer.timesteps_per_batch=4096 trainer.total_timesteps=15000000 \
    trainer.num_minibatches=32 trainer.n_updates_per_iteration=4 robot.v_max=1.2 "$@"
  printf '\n'
}

command -v sbatch >/dev/null || { printf 'sbatch is required on Matilda.\n' >&2; exit 2; }
mkdir -p "$root/outputs/log"
for group in "${groups[@]}"; do
  submit=(sbatch --parsable "--job-name=diffrl-$group" --account=scale-lab
    --partition=gpu6000 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=8
    --mem=64G --gres=gpu:6000:1 --time=3-00:00:00 --export=ALL
    "--chdir=$root" "--output=$root/outputs/log/slurm-%j.out")
  printf '\n[%s] 1 GPU, 8 CPUs\n' "$group"
  job_script "$@" | "${submit[@]}"
done
