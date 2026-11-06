# Verified project portfolio

This page is the concise entry point for four runnable, evidence-backed projects.
Each project preserves its original root files and provides a clean-archive
verification command, architecture notes, measured results and explicit limits.
The revision links below are evidence anchors verified through November 6, 2026;
`main` may continue to evolve.

The canonical [`portfolio.json`](portfolio.json) companion is checked against
this page with `python3 scripts/validate_portfolio.py`. The validator enforces a
strict schema, unique repositories and revisions, exact GitHub URL construction,
canonical JSON, and synchronized commands, results and limitations.

The dated [evidence-health audit](EVIDENCE_HEALTH.md) records the commit objects,
current showcase paths and push-triggered CI runs observed on November 6, 2026.
Its validator is offline; the snapshot never turns normal verification into a
network-dependent check.

The [release checklist](RELEASE_CHECKLIST.md) cross-validates that every public
entry exposes setup, demo, checks, pinned evidence and an adjacent limitation.
The canonical [`release_summary.json`](release_summary.json) binds all three
machine-readable artifacts to their validators and exact SHA-256 digests. Run
`python3 scripts/verify_portfolio_release.py` for the complete offline release
check.

## Project map

| Project | Demonstrates | Clean verification | Evidence anchor |
| --- | --- | --- | --- |
| [C++ autograd engine](https://github.com/BLANK-2340/cpp-autograd-engine) | Header-only scalar reverse-mode autodiff, deterministic MLP/SGD, dense tensor gradients, stable multiclass loss and versioned checkpoints | `bash scripts/verify_clean_checkout.sh` | [`32e5cdf`](https://github.com/BLANK-2340/cpp-autograd-engine/commit/32e5cdf5a50f91ce12bd187a15b9ab4660cdedf0) |
| [VisionTrack-NN](https://github.com/BLANK-2340/VisionTrack-NN) | Dependency-light NumPy training, checkpoint/image validation, calibration and controlled Fashion-MNIST subset comparisons | `./scripts/verify_showcase.sh` | [`1267475`](https://github.com/BLANK-2340/VisionTrack-NN/commit/12674754043ee6e7b097a549078736b5f913be01) |
| [CUDA kernel lab](https://github.com/BLANK-2340/cuda-kernel-lab) | Checked C++ CPU oracles, caller-owned buffers, hash-bound evidence and reproducibility receipts, normalized environment metadata and optional CUDA kernels | `./scripts/verify_showcase.sh` | [`ce99741`](https://github.com/BLANK-2340/cuda-kernel-lab/commit/ce9974174983087641d3f814eab139d451f9101f) |
| [Pendulum wave simulation](https://github.com/BLANK-2340/Pendulum-Wave-simulation) | Deterministic nonlinear RK4 simulation, CSV/GIF outputs, convergence evidence and bounded energy-drift measurements | `./scripts/verify_showcase.sh` | [`24be3e3`](https://github.com/BLANK-2340/Pendulum-Wave-simulation/commit/24be3e3b53ff69435bae32c1e4563ab7f8ebb909) |

Run each command from that repository's root after following its showcase setup.
The verifiers export committed `HEAD` to a temporary directory, so success does
not depend on untracked files or an existing build directory.

## Recorded results and scope

### C++ autograd engine

- Scalar demo: `f=15`, `df/dx=8`, `df/dy=3` for
  `f=x*x+x*y`, `x=3`, `y=2`.
- XOR seed 42: MSE `0.000218520` and accuracy `4/4` after 4,000 updates.
- Synthetic multiclass example: cross-entropy `1.098612289 -> 0.009116757`,
  training accuracy `12/12`, disjoint holdout accuracy `6/6`.
- Checkpointed 120+180-step continuation exactly matched the uninterrupted
  300-step trajectory in the same build.
- [Architecture, commands and narrower limits](https://github.com/BLANK-2340/cpp-autograd-engine/blob/main/docs/SHOWCASE.md)

Scope: dependency-free educational CPU `double` arithmetic. The synthetic data
does not establish real-world generalization, throughput or competitive accuracy.

### VisionTrack-NN

- Synthetic orientation task: loss `1.108148 -> 0.011034`; train and independent
  generated-test accuracy `1.0000`.
- Three-seed spatial Fashion-MNIST subset suite: mean accuracy `72.67%`, sample
  standard deviation `0.70` percentage points, mean macro F1 `0.723750`.
- Equal-parameter grid/global comparison on the same fixed subset: mean accuracy
  `72.67%` versus `36.00%`; paired difference `+36.67` percentage points.
- [Architecture, protocols and narrower limits](https://github.com/BLANK-2340/VisionTrack-NN/blob/main/docs/SHOWCASE.md)

Scope: Fashion-MNIST figures use fixed balanced subsets, not the full official
train/test sets. The comparison is parameter matched but not compute matched and
is not a pooling-only causal ablation. No GPU, latency or state-of-the-art claim
is made.

### CUDA kernel lab

- Six-process Linux/x86-64 CPU snapshot with GCC 13.3.0:
  vector addition `0.077863 +/- 0.001668 ms` allocating and
  `0.027052 +/- 0.000423 ms` preallocated.
- Reduction: `0.045291 +/- 0.000868 ms` allocating and
  `0.043788 +/- 0.000862 ms` preallocated.
- Matrix multiplication: `0.203346 +/- 0.018275 ms` allocating and
  `0.177501 +/- 0.014039 ms` preallocated.
- The pinned revision validates a hash-bound CPU evidence manifest, a
  clean-checkout reproducibility receipt and saved environment fingerprints
  without executing compiler probes during offline validation.
- [Methodology, commands and narrower limits](https://github.com/BLANK-2340/cuda-kernel-lab/blob/main/docs/SHOWCASE.md)

Scope: allocating and preallocated modes measure different timing boundaries;
their ratios are not kernel speedups. Environment fingerprints are descriptive
metadata, not hardware authentication or portable performance evidence. NVCC
compilation, GPU execution, compute-sanitizer, GPU correctness and GPU timing
remain pending actual NVIDIA runtime access.

### Pendulum wave simulation

- Four-second final-state refinement observed RK4 orders `4.026925`, `4.015949`
  and `4.008923` against a finer numerical reference.
- Over one 32-second undamped trajectory, maximum relative energy drift decreased
  from `2.377632494744e-4` at `dt=1/30 s` to `7.319656431532e-9` at
  `dt=1/240 s`.
- The canonical evidence manifest binds both reports to their experiment
  identities, validation commands, measured scopes and SHA-256 hashes.
- [Architecture, commands and narrower limits](https://github.com/BLANK-2340/Pendulum-Wave-simulation/blob/main/docs/SHOWCASE.md)

Scope: the fine-step trajectory is a numerical reference, not an analytical
solution. RK4 is not symplectic; one fixed-duration undamped run does not prove
exact or bounded long-time conservation. No apparatus or performance claim is
made.

## Suggested review order

1. Start with the C++ autograd scalar and XOR demos for algorithms and gradient
   verification.
2. Review VisionTrack-NN for dependency-light experiment design, data provenance
   and careful result interpretation.
3. Review the CUDA lab for benchmark-scope discipline and CPU/GPU separation.
4. Finish with the pendulum project for deterministic simulation and
   machine-verifiable numerical evidence.

The conditional `tiny-transformer-lab`, `vision-inference-bench` and
`paper-to-code-lab` repositories are not included because their exact existence
and accessibility have not been confirmed. No language-model or paper-
reproduction milestone is claimed here.
