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
The final package is installed at `/home/daiyp/CODE/diff_rl_safe`, with its own
`.venv`; tests run from that location. Test outputs remain in temporary paths.
The six-model Hydra configuration resolves successfully from the new root.
