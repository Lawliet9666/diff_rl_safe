#!/usr/bin/env bash
# Temporary plan based on outputs/wandb_summary_20261101/index.html.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_matilda_remaining_tmp.sh [--ppo-filter-only] [--after-ppo-job ID] [Hydra overrides]

Submits three independent jobs, each with 1 GPU, 8 CPUs and 64 GB:
  cbf_rl:         all five robots, using the updated CBF-RL reward
  barriernet:     double_integrator, humanoid, quadrotor
  barriernet_sim: double_integrator, humanoid, quadrotor
Robots run sequentially within each job. Each training starts from zero
with the usual Matilda configuration: 32 environments, batch 4096, 15M steps.

PPO+filter is left to job 404073 by default. --ppo-filter-only submits just
one job for unicycle, double_integrator, humanoid and quadrotor instead.
--after-ppo-job ID selects that PPO+filter job and waits for ID to finish
with any exit status, allowing the current robot to finish before the handoff.
Submission checks squeue and stops if a matching job is still active.
Extra Hydra overrides take precedence; authenticate W&B before submission.
HELP
}

ppo_only=false
ppo_after=""
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ppo-filter-only) ppo_only=true; shift ;;
    --after-ppo-job)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' '--after-ppo-job requires a positive numeric job ID.' >&2; exit 2
      fi
      ppo_only=true; ppo_after="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; overrides+=("$@"); break ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    *) overrides+=("$1"); shift ;;
  esac
done
# Preserve an empty array on macOS Bash 3.2 as well as the HPC runtime.
set -- ${overrides[@]+"${overrides[@]}"}
for argument in "$@"; do
  case "$argument" in
    robot=*|+robot=*|++robot=*|device=*|+device=*|++device=*|run_name=*|+run_name=*|++run_name=*)
      printf 'Launcher controls robot/device/run names; remove %s (use RUN_NAME for a prefix).\n' "$argument" >&2
      exit 2 ;;
  esac
done

root="$(cd "$(dirname "$0")/.." && pwd)"
groups=(cbf_rl barriernet barriernet_sim)
if "$ppo_only"; then groups=(ppo_filter); fi

command -v sbatch >/dev/null || { printf 'sbatch is required on Matilda.\n' >&2; exit 2; }
command -v squeue >/dev/null || { printf 'squeue is required to check existing jobs.\n' >&2; exit 2; }
queue="$(squeue --noheader --user "${USER:?USER must be set}" --format='%A %j')"
for group in "${groups[@]}"; do
  case "$group" in
    cbf_rl) previous=404074 ;;
    barriernet) previous=404075 ;;
    barriernet_sim) previous=404076 ;;
    ppo_filter) previous=404073 ;;
  esac
  while read -r job_id job_name; do
    if [[ "$group" == ppo_filter && -n "$ppo_after" && "$job_id" == "$ppo_after" ]]; then
      continue  # This job is an explicit dependency, not a duplicate submission.
    fi
    if [[ "$job_id" == "$previous" || "$job_name" == "diffrl-$group" || "$job_name" == "diffrl-left-$group" ||
          ( "$group" == ppo_filter && "$job_name" == diffrl-ppo_both ) ]]; then
      printf 'Job %s (%s) is still active; inspect it before submitting %s again.\n' "$job_id" "$job_name" "$group" >&2
      exit 2
    fi
  done <<< "$queue"
done
mkdir -p "$root/outputs/log"

job_script() {
  local group="$1"
  shift
  local robot
  local robots=()
  case "$group" in
    cbf_rl) robots=(single_integrator unicycle double_integrator humanoid quadrotor) ;;
    barriernet|barriernet_sim) robots=(double_integrator humanoid quadrotor) ;;
    ppo_filter) robots=(unicycle double_integrator humanoid quadrotor) ;;
  esac
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
if os.path.realpath(sys.prefix) != os.path.realpath(os.environ["CONDA_PREFIX"]):
    raise RuntimeError(f"Wrong Python: {sys.executable}")
CHECK_PYTHON

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-xinywa_umich}"
export WANDB_PROJECT="${WANDB_PROJECT:-diff_rl_safe}"
export RUN_NAME="${RUN_NAME:-matilda_left_20261101}_${SLURM_JOB_ID}"
export CPU_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/scratch/users/${USER}/tmp/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
SLURM
  for robot in "${robots[@]}"; do
    printf '%q ' bash "$root/scripts/run_local.sh" "$group" "$robot" --device cuda \
      trainer.num_envs=32 trainer.timesteps_per_batch=4096 trainer.total_timesteps=15000000 \
      trainer.num_minibatches=32 trainer.n_updates_per_iteration=4 robot.v_max=1.2 "$@"
    printf '\n'
  done
}

for group in "${groups[@]}"; do
  submit=(sbatch --parsable "--job-name=diffrl-left-$group" --account=scale-lab
    --partition=gpu6000 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=8
    --mem=64G --gres=gpu:6000:1 --time=3-00:00:00 --export=ALL
    "--chdir=$root" "--output=$root/outputs/log/slurm-%j-remaining.out")
  if [[ "$group" == ppo_filter && -n "$ppo_after" ]]; then
    submit+=("--dependency=afterany:$ppo_after")
  fi
  printf '\n[%s] 1 GPU, 8 CPUs; fresh training runs\n' "$group"
  job_script "$group" "$@" | "${submit[@]}"
done
