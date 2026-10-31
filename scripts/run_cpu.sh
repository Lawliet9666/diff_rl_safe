#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cpu.sh [--dry-run] METHOD ROBOT [Hydra overrides / controller flags]
  bash scripts/run_cpu.sh [--dry-run] eval RUN_DIR [evaluation flags]

Methods (GNN policies by default):
  ppo             Ordinary PPO, without a safety filter.
  ppo_filter      PPO with a fixed post-sampling CBF-QP filter.
  barriernet      PPO with a learned differentiable CBF-QP layer.
  barriernet_sim  BarrierNet plus simulation reward gradients (weight=0.1, horizon=8).
  cbf             Goal-tracking CBF-QP; no training (slack=10, nearest 5 of 30 humans).

Robots: single_integrator, unicycle, double_integrator, humanoid, quadrotor.
RL defaults: 4 environments, batch 1024, target 15,000,000 steps; 30 humans; W&B online.
CPU_THREADS defaults to 4; thread settings do not enforce CPU affinity.
RUN_NAME, WANDB_MODE, WANDB_ENTITY and WANDB_PROJECT retain their existing meanings.
Extra Hydra overrides take precedence over the profile defaults.

Examples:
  bash scripts/run_cpu.sh --dry-run barriernet unicycle
  bash scripts/run_cpu.sh ppo_filter humanoid seed=0
  bash scripts/run_cpu.sh barriernet_sim quadrotor trainer.reward_loss.weight=0.05
  bash scripts/run_cpu.sh cbf unicycle controller.slack_weight=0 --visualize
  bash scripts/run_cpu.sh eval outputs/crowd_dyn_var_num_env/runs/RUN --episodes 1 --visualize
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
method="$1"
target="$2"
shift 2
threads="${CPU_THREADS:-4}"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CPU_THREADS must be a positive integer.\n' >&2
  exit 2
fi

cd "$(dirname "$0")/.."
if [[ "$method" != "eval" ]]; then
  case "$target" in
    single_integrator|unicycle|double_integrator|humanoid|quadrotor) ;;
    *) printf 'Unknown robot: %s\n' "$target" >&2; exit 2 ;;
  esac
fi
case "$method" in
  ppo) profile=(model=ppo_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.0) ;;
  ppo_filter) profile=(model=ppo_gnn trainer.safety_filter.enabled=true trainer.reward_loss.weight=0.0) ;;
  barriernet) profile=(model=barriernet_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.0) ;;
  barriernet_sim) profile=(model=barriernet_gnn trainer.safety_filter.enabled=false trainer.reward_loss.weight=0.1 trainer.reward_loss.horizon=8) ;;
  cbf|eval) ;;
  *) printf 'Unknown method: %s\n' "$method" >&2; exit 2 ;;
esac

cpu=(env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "OMP_NUM_THREADS=$threads"
  "OPENBLAS_NUM_THREADS=$threads" "VECLIB_MAXIMUM_THREADS=$threads"
  "TF_NUM_INTRAOP_THREADS=$threads" TF_NUM_INTEROP_THREADS=1
  "XLA_FLAGS=${XLA_FLAGS:---xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$threads}")
case "$method" in
  cbf)
    command=(python scripts/run_controller.py "robot=$target" device=cpu "$@") ;;
  eval)
    command=(python scripts/eval.py --save-dir "$target" --device cpu "$@") ;;
  *)
    cpu+=("RUN_NAME=${RUN_NAME:-cpu_${target}_${method}_$(date +%Y%m%d_%H%M%S)}"
      "WANDB_MODE=${WANDB_MODE:-online}")
    command=(bash scripts/run_ppo.sh device=cpu env=crowd_dyn_var_num_env "robot=$target"
      trainer.num_envs=4 trainer.timesteps_per_batch=1024 trainer.total_timesteps=15000000
      "${profile[@]}" "$@") ;;
esac

if "$dry_run"; then
  printf '%q ' "${cpu[@]}" "${command[@]}"
  printf '\n'
else
  exec "${cpu[@]}" "${command[@]}"
fi
