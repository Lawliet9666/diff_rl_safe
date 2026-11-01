#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exec bash "$(dirname "$0")/run_local.sh" cbf "$@"
fi
profile=()
