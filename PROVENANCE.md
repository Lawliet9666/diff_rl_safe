# Source provenance

- Source repository: https://github.com/Lawliet9666/crowd_env
- Source branch: `jax_clean_v2`
- Committed snapshot: `1673dfaa0c72a749c04929145fede48b058eb157`
- Migration date: 2026-10-10
- New history: independent repository, initialized on `codex/jax-baseline` and
  renamed to `main` for the Safety-Gymnasium integration.

Copied from the committed snapshot: six policies and shared helpers, dense JAX
QP solver/custom VJP, PPO training and rollout code, JAX crowd environment,
checkpoint/device helpers, required Hydra configs, training launcher, and the
ordinary rollout renderer. Shared implementations were extracted without
changing the parameter trees or random-key allocation.

Changes: BarrierNet-GNN default; project/environment name `diff_rl_safe`;
shared MLP, observation, safety-head and QP helpers; robot validation moved
out of environment imports; inherited crowd settings merged into one config;
local training metrics; compact checkpoint evaluator; video capture from
the numerical evaluation itself; complete CPU dependency specifications;
focused documentation and tests.

Excluded: source Git history, historical data/outputs/checkpoints/videos,
analysis scripts, hand-written controllers and their evaluators, unrelated
uncommitted Astra/image-navigation work, unused experiment configs, editor
settings, and advanced diagnostic dashboards.

No source checkout files were edited. The source snapshot has no top-level
license file; this migration assigns no new license to inherited code.

## Retired native Point backend — 2026-10-18

The Safety-Gymnasium host-rollout backend, installer, dedicated dependency list
and tests were removed at the user's request. The pure JAX and CRAX/MJX tasks
remain. Existing run artifacts and installed Micromamba environments were
preserved; no installed packages were removed.

## Optional GPU runtime

`requirements-gpu.txt` adds JAX CUDA plugin/PJRT 0.4.35 and pinned NVIDIA CUDA
12.6 libraries to `diffrl`, preserving JAX/JAXlib 0.4.35 and NumPy 1.26.4.
The source for plugin dependencies is the release packaging at
https://github.com/jax-ml/jax/blob/jax-v0.4.35/jax_plugins/cuda/plugin_setup.py.
The x86_64 CPython 3.10 plugin wheel has SHA256
`5fcc47ef49cb8ff5ceaa67d143a18b22c78456a0967d243963312901d978e96e`.
Installed libraries are environment-local; no system CUDA, driver, or existing
crowd environment is changed.

## Optional CRAX MJX integration (2026-10-17)

Upstream: https://github.com/TTomilin/CRAX (Apache-2.0), pinned commit
`7c12acd8d183810ca896038a3e85d3fe06bece79`. Installed as an optional package
in the separate Python 3.11 `diffrl_crax` environment. No CRAX repository or
training algorithm is vendored or imported. The built wheel SHA256 on this host
was `3660537f9b9d2b4ab3a853c6bbc7574475b3fafffc9587baf8b6b2a3050096d2`.

CRAX provides the Point asset, scene builder, cylindrical collision extensions,
and MJX pipeline. Our functional task retains the crowd, observation and reward
functions. Shared crowd kernels and timeout-aware GAE were extracted from the
native backend without changing their arithmetic. The rollout collector gained
explicit reset/step/observation hooks; its original defaults remain unchanged.

The optional `env/mjx_precision.py` compatibility shim adapts only the physics
modules' local array constructors to float32. MJX 3.11 assumes JAX's default
float32, while our differentiable QP requires x64 to remain enabled. Policies,
solver dtypes and JAX's global NumPy namespace are unchanged. Installed upstream
source files are not patched. Physics-leaf dtype and native-engine parity are
covered by tests. CUDA libraries are pinned to versions compatible with this
host's driver; JAX 0.6.0 and MuJoCo/MJX 3.11.0 match the CRAX stack.

See `requirements-crax-lock.txt`, `environment-crax.yml` and `docs/CRAX.md`.
Existing `diffrl`, crowd environments and original source checkout are preserved.

## Learned-distance BarrierNet extension — 2026-10-18

BarrierNet-GNN and BarrierNet-MLP now optionally learn a bounded per-obstacle
safety distance in addition to α. The distance helper and bounds are shared
with DiffCVaR; deterministic constraints remain separate from CVaR constraints.
New model configs enable the head. Fixed-distance mode and old resolved configs
retain their parameter trees and initialization keys for checkpoint evaluation.
Existing experimental results describe the fixed-distance variant. This is a
method extension after the original numerical-preserving repository migration.
