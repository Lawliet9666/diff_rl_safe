#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_gpu.sh METHOD ROBOT [--gpu ID] [--dry-run] [overrides / controller flags]

Methods: ppo, ppo_filter, barriernet, barriernet_sim, cbf, all.
Robots: single_integrator, unicycle, double_integrator, humanoid, quadrotor, all.
'all all' runs CBF, PPO, PPO+filter, BarrierNet, then BarrierNet+simulation;
each method runs the five robots sequentially. No jobs run in parallel.

Requires CUDA-enabled JAX and exactly one visible GPU. --gpu selects visibility,
not allocation. Omit --gpu to preserve a scheduler's CUDA_VISIBLE_DEVICES.
RL defaults: GNN, 16 environments, batch 4096, target 15,000,000 steps per robot;
30 humans; W&B online. BarrierNet+simulation uses reward weight 0.1 and horizon 8.
CPU_THREADS defaults to 12; thread settings do not reserve cores or enforce affinity.
CBF defaults: slack 10, nearest 5 of 30 humans, one episode per robot; CPU OSQP solve.
RUN_NAME is a prefix; robot/method/time are appended. WANDB_* variables are reused.
Extra Hydra overrides take precedence; robot/device/run names are launcher-controlled.
For METHOD=all, extra arguments apply only to RL; configure CBF separately.

Examples:
  bash scripts/run_gpu.sh all all --gpu 0 --dry-run
  bash scripts/run_gpu.sh all all --gpu 0
  bash scripts/run_gpu.sh barriernet all --gpu 0 seed=1
  bash scripts/run_gpu.sh barriernet unicycle --gpu 0 model=barriernet_mlp
  bash scripts/run_gpu.sh cbf all --gpu 0 --episodes 50 --seeds 100,200
HELP
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
method="$1"
target="$2"
shift 2
case "$method" in
  ppo) profile=(model=ppo_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.0) ;;
  ppo_filter) profile=(model=ppo_gnn trainer.safety_filter.enabled=true trainer.reward_loss.weight=0.0) ;;
  barriernet) profile=(model=barriernet_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.0) ;;
  barriernet_sim) profile=(model=barriernet_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.1 trainer.reward_loss.horizon=8) ;;
  cbf|all) ;;
  *) printf 'Unknown method: %s\n' "$method" >&2; exit 2 ;;
esac
case "$target" in
  single_integrator|unicycle|double_integrator|humanoid|quadrotor|all) ;;
  *) printf 'Unknown robot: %s\n' "$target" >&2; exit 2 ;;
esac

dry_run=false
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --gpu)
      if [[ $# -lt 2 || ! "$2" =~ ^[A-Za-z0-9_./-]+$ || "$2" == -* ]]; then
        printf '%s\n' '--gpu requires one GPU index or UUID.' >&2
        exit 2
      fi
      gpu="$2"
      shift 2 ;;
    *) break ;;
  esac
done
for argument in "$@"; do
  case "$argument" in
    robot=*|+robot=*|++robot=*|device=*|+device=*|++device=*|--device|--device=*|run_name=*|+run_name=*|++run_name=*)
      printf 'Suite controls robot/device/run names; remove %s (use RUN_NAME for a prefix).\n' "$argument" >&2
      exit 2 ;;
  esac
done

threads="${CPU_THREADS-12}"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CPU_THREADS must be a positive integer.\n' >&2
  exit 2
fi

if [[ "$method" == all ]]; then
  for next_method in cbf ppo ppo_filter barriernet barriernet_sim; do
    command=(bash "$0" "$next_method" "$target")
    if [[ -n "$gpu" ]]; then command+=(--gpu "$gpu"); fi
    if "$dry_run"; then command+=(--dry-run); fi
    if [[ "$next_method" != cbf ]]; then command+=("$@"); fi
    "${command[@]}"
  done
  exit 0
fi

cd "$(dirname "$0")/.."
runtime=(env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  "OMP_NUM_THREADS=$threads" "OPENBLAS_NUM_THREADS=$threads" "MKL_NUM_THREADS=$threads"
  "VECLIB_MAXIMUM_THREADS=$threads" "TF_NUM_INTRAOP_THREADS=$threads" TF_NUM_INTEROP_THREADS=1)
if [[ -n "$gpu" ]]; then
  runtime+=("CUDA_VISIBLE_DEVICES=$gpu")
fi
if ! "$dry_run"; then
  "${runtime[@]}" python - <<'PY'
from utils import configure_jax_platform_for_device, validate_jax_backend_for_device
configure_jax_platform_for_device("cuda")
import jax
validate_jax_backend_for_device(jax, "cuda")
devices = jax.devices()
if len(devices) != 1:
    raise RuntimeError(f"Expected one visible GPU, got {devices}. Select --gpu ID or request one GPU from your scheduler.")
print(f"Using {devices[0]}", flush=True)
PY
fi

stamp="$(date +%Y%m%d_%H%M%S)_$$"
robots=("$target")
if [[ "$target" == all ]]; then
  robots=(single_integrator unicycle double_integrator humanoid quadrotor)
fi
for robot in "${robots[@]}"; do
  if [[ "$method" == cbf ]]; then
    command=(python scripts/run_controller.py "robot=$robot" device=cuda "$@")
    run_environment=("${runtime[@]}")
  else
    run_environment=("${runtime[@]}" "WANDB_MODE=${WANDB_MODE:-online}"
      "RUN_NAME=${RUN_NAME:-gpu}_${method}_${robot}_${stamp}")
    command=(bash scripts/run_ppo.sh device=cuda env=crowd_dyn_var_num_env "robot=$robot"
      trainer.num_envs=16 trainer.timesteps_per_batch=4096 trainer.total_timesteps=15000000
      "${profile[@]}" "$@")
  fi
  printf '\n[%s] %s\n' "$method" "$robot"
  if "$dry_run"; then
    printf '%q ' "${run_environment[@]}" "${command[@]}"
    printf '\n'
  else
    "${run_environment[@]}" "${command[@]}"
  fi
done
