#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exec bash "$(dirname "$0")/run_local.sh" cbf_rl "$@"
fi
profile=(model=ppo_gnn trainer.physical_limits.enabled=false trainer.safety_filter.enabled=true
  trainer.cbf_rl.enabled=true trainer.reward_loss.weight=0.0)
