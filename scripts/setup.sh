#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/setup.sh [--device cpu|cuda]

Install and check diffrl; default: cpu. No training or W&B login is started.
CUDA requires Linux and a visible NVIDIA GPU with a working driver.
MAMBA_ROOT_PREFIX selects the environment root (default: ~/micromamba).
MAMBA_EXE optionally selects the Micromamba executable/install location.
Existing environments are verified, not overwritten. Shell profiles are unchanged.
HELP
}
fail() { printf 'Setup failed: %s\n' "$*" >&2; exit 1; }
device=cpu
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || fail '--device requires cpu or cuda'
      device="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
case "$device" in cpu|cuda) ;; *) fail '--device must be cpu or cuda' ;; esac
system="$(uname -s)"
case "$system:$(uname -m)" in
  Darwin:arm64) platform=osx-arm64 ;;
  Darwin:x86_64) platform=osx-64 ;;
  Linux:x86_64) platform=linux-64 ;;
  Linux:aarch64|Linux:arm64) platform=linux-aarch64 ;;
  *) fail 'Unsupported platform; use macOS, Linux, or Windows through WSL' ;;
esac
if [[ "$device" == cuda ]]; then
  [[ "$system" == Linux ]] || fail 'CUDA setup requires Linux (Windows: use WSL)'
  command -v nvidia-smi >/dev/null || fail 'NVIDIA driver/nvidia-smi not found'
  nvidia-smi -L >/dev/null || fail 'No accessible NVIDIA GPU; on HPC, allocate a GPU first'
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
mamba_root="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
[[ "$mamba_root" == /* ]] || fail 'MAMBA_ROOT_PREFIX must be an absolute path'
prefix="$mamba_root/envs/diffrl"
mamba="${MAMBA_EXE:-$(command -v micromamba || true)}"
mamba="${mamba:-$HOME/.local/bin/micromamba}"
[[ "$mamba" == /* ]] || fail 'MAMBA_EXE must be an absolute path'
if [[ ! -x "$mamba" ]]; then
  [[ ! -e "$mamba" ]] || fail "Micromamba is not executable: $mamba"
  command -v curl >/dev/null || fail 'curl is required to install Micromamba'
  bootstrap_dir="$(mktemp -d "${TMPDIR:-/tmp}/diffrl-setup.XXXXXX")"
  trap 'rm -rf "$bootstrap_dir"' EXIT
  printf 'Installing Micromamba 2.4.0 (%s)...\n' "$platform"
  curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 \
    "https://micro.mamba.pm/api/micromamba/$platform/2.4.0" \
    --output "$bootstrap_dir/micromamba.tar.bz2"
  tar -xjf "$bootstrap_dir/micromamba.tar.bz2" -C "$bootstrap_dir" bin/micromamba
  [[ "$("$bootstrap_dir/bin/micromamba" --version)" == 2.4.0 ]] || fail 'Unexpected Micromamba version'
  mkdir -p "$(dirname "$mamba")"
  install -m 755 "$bootstrap_dir/bin/micromamba" "$mamba"
fi
manager=("$mamba" --no-rc --root-prefix "$mamba_root")
if [[ -e "$prefix" ]]; then
  [[ -f "$prefix/conda-meta/history" ]] || fail "Existing path is not a Conda environment: $prefix"
  printf 'Checking existing environment: %s\n' "$prefix"
else
  "${manager[@]}" create --yes --prefix "$prefix" --file "$root/environment.yml"
fi
python_command=("${manager[@]}" run --prefix "$prefix" python)
# Read the JAX version from the installed, YAML-validated environment.
jax_version="$("${python_command[@]}" - "$root/environment.yml" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
import yaml
from packaging.requirements import Requirement

spec = yaml.safe_load(Path(sys.argv[1]).read_text())
python_pin = next(entry.split('=', 1)[1] for entry in spec['dependencies']
                  if isinstance(entry, str) and entry.startswith('python='))
if tuple(sys.version_info[:2]) != tuple(map(int, python_pin.split('.'))):
    raise SystemExit(f'Python {python_pin} is required by environment.yml')
requirements = []
for entry in spec['dependencies']:
    if isinstance(entry, dict):
        requirements.extend(entry.get('pip', []))
    elif not entry.startswith('python='):
        requirements.append(entry)
for entry in requirements:
    requirement = Requirement(entry)
    installed = version(requirement.name)
    if requirement.marker is None or requirement.marker.evaluate():
        if installed not in requirement.specifier:
            raise SystemExit(f'{requirement.name} {installed} does not satisfy {requirement.specifier}; environment.yml must be installed in a compatible environment')
print(version('jax'))
PY
)"
if [[ "$device" == cuda ]]; then
  "${python_command[@]}" -m pip install "jax[cuda12]==$jax_version"
fi
"${python_command[@]}" -m pip check
cd "$root"
printf 'Checking imports, %s JAX, gradients, and CBF-QP...\n' "$device"
env JAX_PLATFORMS="$device" JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  "${python_command[@]}" - "$device" <<'PY'
import importlib
import sys
from utils import configure_jax_platform_for_device, validate_jax_backend_for_device

configure_jax_platform_for_device(sys.argv[1])
import jax
import jax.numpy as jnp
import numpy as np

validate_jax_backend_for_device(jax, sys.argv[1])
for name in ('hydra', 'matplotlib', 'imageio_ffmpeg', 'optax', 'wandb', 'osqp',
             'model.ppo_gnn', 'model.ppo_mlp', 'model.barriernet_gnn', 'model.barriernet_mlp'):
    importlib.import_module(name)
np.testing.assert_allclose(jax.jit(jax.grad(lambda x: jnp.sum(x*x)))(jnp.array([1., 2.])), [2., 4.])

import scripts.run_ppo_base
from hydra import compose, initialize_config_dir
from controller import make_controller
from env.robots.base import Obstacles
from pathlib import Path

with initialize_config_dir(version_base=None, config_dir=str(Path.cwd()/'config')):
    config = compose(config_name='config_controller', overrides=['robot=unicycle'])
robot, _, action_fn = make_controller(config)
x = robot.initial_state(jnp.zeros(3))
obstacles = Obstacles(jnp.array([[2., 1.]]), jnp.zeros((1, 2)), jnp.array([.4]), jnp.ones(1))
action, diagnostics = action_fn(x, robot.nominal_control(x, jnp.array([4., 0.])), obstacles)
assert np.isfinite(action).all() and diagnostics['qp'].valid
print(f'Passed: backend={jax.default_backend()}, devices={jax.devices()}, CBF-QP action={np.asarray(action)}')
PY
printf '\nReady: %s (%s). No training started.\n' "$prefix" "$device"
printf 'Optional W&B login (your own account):\n'
printf '%q ' "$mamba" run --prefix "$prefix" wandb login
printf '\nRun: bash scripts/run_local.sh cbf unicycle --device %s --visualize\n' "$device"
