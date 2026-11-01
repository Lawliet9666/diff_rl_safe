#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  variant="${1:-both}"
  if [[ $# -gt 0 ]]; then shift; fi
  if [[ "$variant" == --help || "$variant" == -h ]]; then exec bash "$(dirname "$0")/run_local.sh" --help; fi
  if [[ "$variant" == both ]]; then variant=ppo_both; fi
  case "$variant" in ppo|ppo_filter|ppo_both) ;; *) printf 'Choose ppo, ppo_filter or both.\n' >&2; exit 2 ;; esac
  exec bash "$(dirname "$0")/run_local.sh" "$variant" "$@"
fi

case "$1" in
  ppo) profile=(model=ppo_gnn trainer.safety_filter.enabled=false) ;;
  ppo_filter) profile=(model=ppo_gnn trainer.safety_filter.enabled=true) ;;
esac
profile+=(trainer.safety_filter.reward.weight=0.0 trainer.reward_loss.weight=0.0)
