# Validation

Validation uses CPU JAX 0.4.35/JAXlib 0.4.35, Python 3.10.16, and the dependencies
in `requirements.txt`. W&B synchronization is disabled throughout verification.

## Source parity

The specified committed snapshot was exported into a separate temporary
reference checkout, excluding uncommitted files. Separate Python processes
tested all six policies with both unicycle and single-integrator dynamics.
Each case used the source default network dimensions, 20 obstacle slots,
masked obstacles, identical keys, and three environment transitions.

Compared 2,172 arrays covering parameter trees, policy outputs, QP context,
diagnostics, rewards, termination flags, metrics, and environment states.
All matched exactly: maximum absolute difference **0.0**. Acceptance tolerances
were `rtol=1e-5`, `atol=1e-6`.

To repeat the probe against two repositories:

```bash
python tests/parity_probe.py <reference-repo> /tmp/reference.pkl
python tests/parity_probe.py <clean-repo> /tmp/clean.pkl
```

The output files contain temporary numerical validation snapshots and should
stay outside the repository. Compare matching dictionary keys and arrays with
NumPy `assert_allclose` using the tolerances above.

## Method tests

The installed repository's initial full suite passed **23 tests**. A subsequent
targeted run passed the added all-masked evaluation case plus all six training
tests (**7 passed**). The final suite contains **24 distinct tests**, all covered by the passing runs.

Method coverage comprises the original **17 passed** checks plus the added
all-masked evaluation check (**18 method tests**).

Coverage: all six policies on both robot types; bounded finite actions and
values; masked obstacles; all-masked slots; changing nearest-obstacle selection;
finite nonzero gradients to BarrierNet-GNN/MLP α heads; a QP active-constraint
solution with known derivative and matching finite difference; parameter
checkpoint round trip.

## Training and video tests

`python -m pytest tests/test_training.py -q`: **6 passed**, one per policy. Each uses
eight environments, two updates, 64 transitions per update, one PPO epoch,
two minibatches, six obstacles, and small networks. Every update checks finite
values and writes a checkpoint. W&B log payloads are captured after initializing
a disabled run and compared with local `metrics.jsonl` records.

BarrierNet-GNN checkpoint evaluation additionally uses two episodes with base
seed 100. Numerical evaluation and rendered evaluation match within the source
parity tolerances. Each MP4 decodes successfully with nine RGB frames at
752×752 resolution. `eval_results.json` is retained when rendering. A final fresh BarrierNet-GNN
training/video test passed after switching decoding to direct FFmpeg
(`1 passed, 5 deselected`), with no fork warning.

## Limits

These checks establish migration parity and working learning/evaluation paths,
not comparative navigation performance or CUDA compatibility.

A constrained slack-QP stress fixture revealed sensitivity to small α changes
under the inherited finite-iteration solver stopping criteria. The committed
source and cleaned implementation return identical values for that fixture.
The solver and its stopping criteria remain unchanged. Full-policy gradient
checks establish finite nonzero α gradients; numerical derivative accuracy is
checked separately on a well-conditioned QP with an analytical solution.

Historical data, checkpoints, outputs, analysis/controller modules, advanced
dashboards, and unrelated uncommitted work are absent from the new repository.
The original source checkout's commit and working-tree status are unchanged.
The final package is installed at `/home/daiyp/CODE/diff_rl_safe`, where the initial checks used a local
`.venv`. That environment was subsequently removed at the user's request;
current runs use the isolated Micromamba `diffrl` environment. Test outputs remain in temporary paths.
The six-model Hydra configuration resolves successfully from the new root.

## Retired native Point backend

The Safety-Gymnasium host-rollout implementation and its dedicated tests were
removed on 2026-10-18. Historical run artifacts and installed environments are
preserved, but that backend is no longer a training or checkpoint-evaluation
option. Current physical-robot checks use CRAX/MJX.

## CRAX/MJX Point crowd integration

The original integration measurements referenced here used the previous
interpolated crowd motion. They are historical rather than measurements of
the robot-first update.

The isolated `diffrl_crax` backend, validation checkpoint, matching videos,
controller/contact tolerances and measured 23× collector speedup are recorded
in [docs/CRAX_VALIDATION.md](docs/CRAX_VALIDATION.md). Setup and runnable commands
are in [docs/CRAX.md](docs/CRAX.md). All six methods completed CPU PPO smoke runs.

## Robot-first crowd update verification

The CRAX backend advances the robot with fixed obstacle positions,
then run the shared goal/social-force/GMM/position update using the actual
robot endpoint. Contacts and minimum clearance include robot physics substeps
and the final crowd placement. The final forward pass advances no time and
applies no new contact impulse.

A before/after JAX probe compared **1,980 arrays** across two robot dynamics,
three crowd configurations and ten transitions per configuration, with three
environments, masked slots, nonzero actions, reactive obstacles, GMM noise and
goal changes. All arrays matched exactly: maximum absolute difference **0.0**.
Temporary reference snapshots remain outside the repository.

Order tests record all 50 robot substeps, verify that active obstacle positions
stay fixed, and check response to the robot's measured endpoint rather than its
old state. Deterministic and GMM/goal-changing scenes cover masks and final
observations. Separate contact scenes verify both an obstacle that approaches
only after robot physics, without moving the robot or advancing extra time,
and a robot-phase contact that must remain recorded after final separation.

CRAX/MJX checks are recorded in
[docs/CRAX_VALIDATION.md](docs/CRAX_VALIDATION.md#robot-first-update-verification).
No new GPU throughput benchmark or avoidance-performance claim is made here.

## Learned-distance BarrierNet verification — 2026-10-18

Both BarrierNet encoders now optionally learn a per-obstacle safety distance
alongside α. New configs enable the distance head with the same 0.75–1.875 m
default range as DiffCVaR. Deterministic CBF constraints, PPO rewards and action
mapping are unchanged. The distance statistic uses the existing local/W&B key
`diagnostics/r_safe_mean`.

CPU verification in `diffrl_crax` passed **43 tests** in 479.18 seconds:
18 new distance tests, the existing 18 method tests, six two-update PPO runs,
and one compiled CRAX Point PPO/checkpoint/numerical-video consistency test.
Tests cover both encoders and both kinematic robot models, distance bounds,
changing nearest obstacles, masks, finite nonzero QP gradients to both α and
distance heads, legacy checkpoint loading, and invalid distance ranges.
The approaching-obstacle gradient fixtures use 1.5 m separation so the new
initial safety margin activates a constraint without saturating the speed bound.
W&B was disabled and test artifacts were written under `/tmp`.

A separate before/after probe captured all parameters and policy outputs before
the model edit for both BarrierNet encoders and both robot dynamics. With old
resolved configs, every parameter and output matched exactly after the edit.
Old configs omit `learn_r_safe` and retain fixed-distance behavior. Existing
20M checkpoints and their reported results remain fixed-distance experiments;
no learned-distance 20M run was launched by this change.

Reproduce the current checks in `diffrl_crax` with command-local CPU/OSMesa
settings:

```bash
WANDB_MODE=disabled JAX_PLATFORMS=cpu JAX_ENABLE_X64=true MUJOCO_GL=osmesa \
  LD_PRELOAD=/data/daiyp/micromamba/envs/diffrl_crax/lib/libOSMesa.so \
  micromamba run -n diffrl_crax python -m pytest -q \
  tests/test_barriernet_distance.py tests/test_methods.py tests/test_training.py \
  tests/test_crax_point.py::test_compiled_autoreset_and_wandb_payload
```

## Pre-commit checks — 2026-10-18

A focused CPU regression run passed **60 tests** in **363.37 seconds**, covering
all six policies, learned-distance gradients, episode/timeout semantics, CRAX
update order/contact cases and native-MuJoCo tracking/task parity. Dependency,
Python syntax and staged whitespace/artifact checks also passed. W&B was
disabled, no GPU was used, and the existing 20M run remained paused.

The contact fixture passed in this grouped run but has an earlier isolated
failure with both forward-pass variants. See
[the forward-cleanup limitation and reproduction command](docs/CRAX_VALIDATION.md#forward-pass-cleanup--2026-10-18).
This check did not repeat the previously recorded PPO/video smoke runs.
