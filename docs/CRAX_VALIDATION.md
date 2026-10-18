# CRAX Point validation — 2026-10-17

Historical report: the original measurements below used crowd motion computed
before robot physics and interpolated through its substeps. Timings, encounter
steps and checkpoint metrics are not measurements of the subsequent robot-first
update. Any robot-first checks are documented separately at the end of this file.

The Safety-Gymnasium host collector was removed on 2026-10-18. Its earlier
measurements below remain historical comparisons; its dedicated tests and
benchmark command are no longer available. Native MuJoCo comparisons using
the CRAX-built model remain supported by the CRAX tests.

Hardware: NVIDIA RTX 6000 Ada (48 GB), driver 570.211.01. CRAX was measured on
GPU 0 and native MuJoCo on the identical GPU 1, with W&B disabled. No online
verification runs were uploaded. Environment: `/data/daiyp/micromamba/envs/diffrl_crax`.

## Implementation checks

- CRAX `7c12acd8d183810ca896038a3e85d3fe06bece79`, Python 3.11, JAX/JAXlib
  0.6.0 and MuJoCo/MJX 3.11.0 installed; `pip check` passes. Exact pip and
  Conda package resolutions are recorded in the repository.
- Unmodified stock CRAX Point reset/step passed JIT and vmap on GPU, with a
  collidable cylindrical obstacle. The custom task's renderer produced playable
  640×640 H.264/yuv420p videos at 10 fps.
- All physics floating leaves, controls and model parameters are float32;
  policy/QP code and parameter structures are unchanged. The precision shim
  changes physics modules' local array constructors; JAX x64 remains enabled.
- Existing JAX regression suite: **24 passed** in `diffrl_crax`, including all
  six policies with both JAX robot dynamics, masks, changing top-k selection,
  QP backward finite differences, and nonzero finite BarrierNet α gradients.
- Existing native Safety-Gymnasium suite: **15 passed** in `diffrl` after the
  shared crowd and timeout-advantage extraction.
- New CRAX tests: **3 passed**, covering task arithmetic/observations, batching,
  masks, steady tracking and braking, native-engine parity, success, collision,
  timeout, compiled fresh-key resets/history and captured W&B metric payloads.
- `model/` and `solver/` have no changes. The original crowd repository, the
  existing `diffrl` environment and its installed packages were preserved.

CPU PPO smoke configuration: 2 environments, 16 transitions/update, 2 updates,
1 PPO epoch, 2 minibatches, 8-step episodes, full configured networks and
observation histories. W&B was disabled. All six methods passed; see `outputs/crax_validation/smokes/` and `smokes.json`.
The full-size GPU verification used 128 environments × 32 steps, batch 4096,
4 configured PPO epochs, 32 minibatches and the existing KL early-stop setting.

## Physics comparisons

The same CRAX-built Point model, controller, 0.002 s timestep and solver were
compared with native **MuJoCo 3.11.0**, independent of the older native collector:

- Same-key initialization and motion match existing JAX functions within
  float32 tolerance `rtol=1e-6, atol=1e-6` (integer fields match exactly).
  Stationary masked-scene rewards and episode counters match the original task.
- One control interval of combined speed/turn motion passed position/yaw
  tolerance `atol=2e-4, rtol=2e-3` and velocity tolerance
  `atol=2e-3, rtol=2e-3`.
- Steady speed tracking error was below 0.06 m/s; braking reduced planar speed
  below 0.01 m/s in the tested scenario.
- Native physical-contact onset: **0.545 m** center separation; MJX: **0.550 m**.
  Sweep resolution was 0.005 m, with an acceptance tolerance of 0.02 m.
- Deliberately unprotected constant commands produced planning collisions at
  steps 16 (head-on), 18 (crossing) and 17 (overtaking). These test moving-crowd
  motion and collision detection; they are not learned-policy performance tests.

The retired collector used MuJoCo **2.3.3** and JAX **0.4.35**, whereas CRAX uses
3.11.0/0.6.0 and its CG solver configuration. MJX uses float32 and approximate
SDF cylinder contacts; native MuJoCo uses float64 and its native collision
routines. These engines are not expected to produce bit-identical long rollouts.
No cylinders were replaced by boxes. Geometric planning collisions and physical
contacts are recorded separately.

## Synchronized full PPO update benchmark

| Collector | Environments × horizon | Warm update | Transitions/s | Peak JAX device buffers |
|---|---:|---:|---:|---:|
| CRAX/MJX | 128 × 32 | 4.37 s | 937.5 | 519.9 MiB |
| Native MuJoCo | 8 × 512 | 99.46 s | 41.2 | 538.1 MiB |

The observed speedup is **22.8×**. CRAX's full PPO-update
compilation took **87.54 s**; model construction and reset compilation are additional.
Three complete updates were timed, synchronizing device work. The first was
excluded from each warm average. Timing excludes evaluation, rendering, W&B and
logging. The benchmark holds learning rate at 1e-4; normal training uses the
configured schedule. Epoch counts can vary through the preserved KL early stop.
The memory figures are JAX device-buffer peaks, not total driver/process memory.

At this measured warm rate, 20 million aggregate steps project to
**5.93 hours** for CRAX and **5.62 days** for the native collector,
before evaluation and startup overhead. This is a short-run estimate, not a
completed 20-million-step experiment. This delivery did not start that full run.

## Delivered checkpoint and matching videos

Run directory:
`outputs/crax_validation/runs/crax_point_acceptance-barriernet_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo`

Checkpoint: `ckpt_00012288.pkl` (**12,288 aggregate steps**, three full-size updates).
The benchmark directory contains resolved configuration, timings, rollout
metrics and checkpoint. Standard trainer smoke directories also contain local
`metrics.jsonl` files with the same metric values sent to W&B.

Held-out seeds: **100 and 200**, one episode each, 400 control steps per episode.
Numerical results: `eval_results.json`; per-episode results and videos are in
`visualize_ckpt_00012288/`:

- `episode_000_seed_100.mp4`: return −1.35194, timeout.
- `episode_001_seed_200.mp4`: return −5.14949, timeout.

Successes: **0/2**; planning collisions: **0/2**; physical contacts: **0/2**;
timeouts: **2/2**. Mean minimum clearance: **0.10293 m**. This verification
checkpoint has not learned goal-reaching behavior and does not establish good
avoidance performance.

Both MP4s decode correctly and contain **401 frames** (initial state plus 400
transitions). NPZ traces preserve the exact qpos/qvel/mocap states used to render
those episodes. Their reward sums match the JSON returns. A separate compiled
evaluation without video matched every numerical metric within `rtol=1e-5,
atol=1e-6`. Rendering replays states; it does not resimulate the actions.

See [CRAX.md](CRAX.md) for installation, training and checkpoint evaluation.

## Robot-first update verification

Subsequent CPU checks use the robot-first crowd update: obstacles remain fixed
through the 50 robot physics substeps, then update from the actual robot
endpoint. A final forward pass records new contacts without advancing time or
applying their impulses. Existing actuator, observation and PPO interfaces are
preserved.

In `diffrl_crax`, the existing method/training/CRAX regression run passed
**27 tests**. This includes all six policies on the original JAX backend,
native-engine physics parity, batched episode events and a compiled
BarrierNet-GNN Point PPO smoke run with two updates and fresh-key auto-resets.

The new update-order suite passed **3 tests**, with three native-backend cases
skipped because Safety-Gymnasium is installed in the separate `diffrl`
environment. Together these runs cover **30 distinct tests** in `diffrl_crax`.
The order suite records each robot substep, verifies endpoint-dependent crowd
motion with and without GMM/goal changes, and covers masked slots. Contact tests
retain robot-phase events after final separation and detect new crowd contacts
at the endpoint without applying an impulse or advancing additional time.

The Point PPO smoke checkpoint reloads through the standard evaluator. Its
numerical-only and rendered metrics match within `rtol=1e-5, atol=1e-6`; the
MP4 decodes successfully, and the per-episode seed and length match the JSON.
W&B uploads were disabled. Test checkpoints, numerical outputs, videos and the
JAX before/after snapshots are in temporary directories outside the repository.

The JAX extraction probe compared **1,980 arrays** before and after the shared
motion extraction, covering both original robot dynamics and three crowd
configurations. Maximum absolute difference was **0.0**. The retired native Point and order
checks separately passed **18 tests** in `diffrl` before their removal.
No new GPU benchmark or learned avoidance-performance result was collected for
this ordering change; the original measurements above remain historical.

## Forward-pass cleanup — 2026-10-18

Only the redundant `mjx.forward` before each robot physics substep was removed.
The refresh after each physics step, all 50 clearance/contact checks, the final
refresh after crowd placement, and reset initialization remain in place.

A direct CPU comparison of three contact scenarios compared all 142 output
leaves before and after the change; they matched exactly. The targeted CPU
checks passed both deterministic/GMM update-order cases and native-engine
tracking/task parity.

An earlier isolated CPU run exposed a contact-test inconsistency in
`tests/test_step_order.py::test_contacts_include_robot_phase_and_final_obstacle_update`.
Its second scene starts with a deep robot/obstacle overlap at 0.4 m center
separation. That run reported a planning collision but missed the expected
transient physical-contact flag. The same assertion failed with the original
pre-substep forward restored in memory, so it predates this cleanup.

The same fixture passed without assertion or collision-logic changes in the
grouped pre-commit run below. These different results suggest sensitivity to
execution context; the cause has not been established. The earlier isolated
failure remains a known validation limitation.

## Pre-commit checks — 2026-10-18

The focused CPU regression run passed **60 tests** in **363.37 seconds**, using
four logical CPU cores, W&B disabled and no GPU. The selection included all
method tests, learned-distance tests, episode/timeout semantics, all three
update-order/contact cases, and native-MuJoCo tracking/task parity:

```bash
WANDB_MODE=disabled JAX_PLATFORMS=cpu JAX_ENABLE_X64=true MUJOCO_GL=osmesa \
  LD_PRELOAD="$CONDA_PREFIX/lib/libOSMesa.so" python -m pytest -q -p no:cacheprovider \
  tests/test_methods.py tests/test_barriernet_distance.py \
  tests/test_episode_semantics.py tests/test_step_order.py \
  tests/test_crax_point.py::test_crax_task_batch_events_tracking_and_native_parity
```

This was a focused regression run; PPO training and video smoke tests were not
repeated. Their prior measurements remain recorded above and in `VALIDATION.md`.
`pip check` passed, all 54 Python source files parsed successfully, and staged
whitespace/artifact checks passed. Generated outputs, W&B data and caches are
ignored. The pip lock now pins `packaging==26.3` instead of its nonportable
Conda build-directory reference. The existing 20M training job remained paused.
