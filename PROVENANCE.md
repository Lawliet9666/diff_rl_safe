# Source provenance

- Source repository: https://github.com/Lawliet9666/crowd_env
- Source branch: `jax_clean_v2`
- Committed snapshot: `1673dfaa0c72a749c04929145fede48b058eb157`
- Migration date: 2026-10-10
- New history: independent repository, branch `codex/jax-baseline`.

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
