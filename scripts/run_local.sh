#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_local.sh METHOD ROBOT [--device cpu|cuda] [--gpu ID] [--dry-run] [overrides]
       bash scripts/run_local.sh eval RUN_DIR [--device cpu|cuda] [--dry-run] [evaluation flags]

Methods: ppo, ppo_filter, ppo_both, cbf_rl, barriernet, barriernet_sim, cbf, all.
Robots: single_integrator, unicycle, double_integrator, humanoid, quadrotor, all.
ppo_both runs PPO then PPO+filter. all runs CBF then the five RL methods.
Methods and robots run sequentially on this machine; Matilda submits separate jobs.

Defaults: CPU, 4 threads, 4 environments, batch 1024; CUDA uses 8 threads,
16 environments and batch 4096. RL targets 15M steps, 30 humans and W&B online.
CPU_THREADS sets thread limits, not CPU affinity. --gpu sets visibility only;
omit it under Slurm to preserve the allocated GPU. Extra overrides take precedence.
RUN_NAME is a prefix; each run appends method, robot and a timestamp.
Humanoid CBF-RL selects the clearance barrier; override controller.humanoid.barrier to change it.
For METHOD=all, overrides apply only to RL; configure CBF separately.
HELP
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
selection="$1"
target="$2"
shift 2
if [[ "$target" == --help || "$target" == -h ]]; then usage; exit 0; fi
device=cpu
gpu=""
dry_run=false
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device|--gpu)
      if [[ $# -lt 2 || "$2" == -* ]]; then printf '%s requires a value.\n' "$1" >&2; exit 2; fi
      if [[ "$1" == --device ]]; then device="$2"; else gpu="$2"; fi
      shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; overrides+=("$@"); break ;;
    *) overrides+=("$1"); shift ;;
  esac
done
# Preserve empty argument lists on the macOS Bash 3.2 runtime.
set -- ${overrides[@]+"${overrides[@]}"}
case "$selection" in
  ppo|ppo_filter|cbf_rl|barriernet|barriernet_sim|cbf) methods=("$selection") ;;
  ppo_both) methods=(ppo ppo_filter) ;;
  all) methods=(cbf ppo ppo_filter cbf_rl barriernet barriernet_sim) ;;
  eval) methods=(eval) ;;
  *) printf 'Unknown method: %s\n' "$selection" >&2; exit 2 ;;
esac
case "$device" in cpu|cuda) ;; *) printf 'Device must be cpu or cuda.\n' >&2; exit 2 ;; esac
if [[ "$selection" != eval ]]; then
  case "$target" in
    single_integrator|unicycle|double_integrator|humanoid|quadrotor|all) ;;
    *) printf 'Unknown robot: %s\n' "$target" >&2; exit 2 ;;
  esac
fi
if [[ -n "$gpu" ]]; then
  if [[ "$device" != cuda || ! "$gpu" =~ ^[A-Za-z0-9_./-]+$ || "$gpu" == -* ]]; then
    printf '%s\n' '--gpu requires CUDA and one GPU index or UUID.' >&2; exit 2
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf 'Slurm manages GPU visibility; omit --gpu.\n' >&2; exit 2
  fi
fi
for argument in "$@"; do
  case "$argument" in
    robot=*|+robot=*|++robot=*|device=*|+device=*|++device=*|run_name=*|+run_name=*|++run_name=*)
      printf 'Launcher controls robot/device/run names; remove %s (use RUN_NAME for a prefix).\n' "$argument" >&2
      exit 2 ;;
  esac
done

scripts="$(cd "$(dirname "$0")" && pwd)"
cd "$scripts/.."
if [[ "$device" == cpu ]]; then
  threads="${CPU_THREADS:-4}"; num_envs=4; batch=1024
else
  threads="${CPU_THREADS:-8}"; num_envs=16; batch=4096
fi
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then printf 'CPU_THREADS must be a positive integer.\n' >&2; exit 2; fi
runtime=(env "JAX_PLATFORMS=$device" JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  "OMP_NUM_THREADS=$threads" "OPENBLAS_NUM_THREADS=$threads" "MKL_NUM_THREADS=$threads"
  "VECLIB_MAXIMUM_THREADS=$threads" "TF_NUM_INTRAOP_THREADS=$threads" TF_NUM_INTEROP_THREADS=1)
if [[ "$device" == cpu ]]; then
  runtime+=("XLA_FLAGS=${XLA_FLAGS:---xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$threads}")
elif [[ -n "$gpu" ]]; then
  runtime+=("CUDA_VISIBLE_DEVICES=$gpu")
fi
if [[ "$device" == cuda ]] && ! "$dry_run"; then
  "${runtime[@]}" python - <<'PY'
from utils import configure_jax_platform_for_device, validate_jax_backend_for_device
configure_jax_platform_for_device("cuda")
import jax
validate_jax_backend_for_device(jax, "cuda")
devices = jax.devices()
if len(devices) != 1:
    raise RuntimeError(f"Expected one visible GPU, got {devices}. Select --gpu ID or request one GPU from Slurm.")
print(f"Using {devices[0]}", flush=True)
PY
fi

stamp="$(date +%Y%m%d_%H%M%S)_$$"
robots=("$target")
if [[ "$target" == all && "$selection" != eval ]]; then
  robots=(single_integrator unicycle double_integrator humanoid quadrotor)
fi
for method in "${methods[@]}"; do
  profile=()
  if [[ "$method" != eval ]]; then
    case "$method" in ppo|ppo_filter) source "$scripts/run_ppo.sh" "$method" ;;
      *) source "$scripts/run_${method}.sh" ;;
    esac
  fi
  for robot in "${robots[@]}"; do
    run_environment=("${runtime[@]}")
    case "$method" in
      cbf)
        command=(python scripts/run_controller.py "robot=$robot" "device=$device")
        if [[ "$selection" != all ]]; then command+=("$@"); fi ;;
      eval)
        command=(python scripts/eval.py --save-dir "$target" --device "$device" "$@") ;;
      *)
        run_environment+=("WANDB_MODE=${WANDB_MODE:-online}")
        command=(python scripts/run_ppo_base.py
          "run_name=${RUN_NAME:-$device}_${method}_${robot}_${stamp}"
          "wandb_entity=${WANDB_ENTITY:-}" "wandb_project=${WANDB_PROJECT:-diff_rl_safe}"
          "device=$device" env=crowd_dyn_var_num_env "robot=$robot"
          "trainer.num_envs=$num_envs" "trainer.timesteps_per_batch=$batch"
          trainer.total_timesteps=15000000 "${profile[@]}")
        if [[ "$method" == cbf_rl && "$robot" == humanoid ]]; then
          command+=(controller.humanoid.barrier=clearance)
        fi
        command+=("$@") ;;
    esac
    printf '\n[%s] %s\n' "$method" "$robot"
    if "$dry_run"; then
      printf '%q ' "${run_environment[@]}" "${command[@]}"; printf '\n'
    else
      "${run_environment[@]}" "${command[@]}"
    fi
  done
done
