#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exec bash "$(dirname "$0")/run_local.sh" barriernet "$@"
fi
profile=(model=barriernet_gnn trainer.safety_filter.enabled=false
  trainer.cbf_rl.enabled=false trainer.reward_loss.weight=0.0)
