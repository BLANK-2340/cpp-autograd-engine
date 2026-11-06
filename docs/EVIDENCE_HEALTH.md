# Portfolio evidence health

This is a dated, read-only audit of the public evidence anchors in
[`portfolio.json`](portfolio.json). It records what GitHub returned on
November 6, 2026; it is a snapshot, not a live availability guarantee.

| Project | Pinned commit | Commit object | Push CI observed |
| --- | --- | --- | --- |
| [C++ autograd engine](https://github.com/BLANK-2340/cpp-autograd-engine/blob/main/docs/SHOWCASE.md) | [`32e5cdf`](https://github.com/BLANK-2340/cpp-autograd-engine/commit/32e5cdf5a50f91ce12bd187a15b9ab4660cdedf0) | reachable, tree `238d9b0` | [C++ correctness: success](https://github.com/BLANK-2340/cpp-autograd-engine/actions/runs/39776734605) |
| [VisionTrack-NN](https://github.com/BLANK-2340/VisionTrack-NN/blob/main/docs/SHOWCASE.md) | [`1267475`](https://github.com/BLANK-2340/VisionTrack-NN/commit/12674754043ee6e7b097a549078736b5f913be01) | reachable, tree `f870226` | [Vision baseline: success](https://github.com/BLANK-2340/VisionTrack-NN/actions/runs/39965186028) |
| [CUDA kernel lab](https://github.com/BLANK-2340/cuda-kernel-lab/blob/main/docs/SHOWCASE.md) | [`ce99741`](https://github.com/BLANK-2340/cuda-kernel-lab/commit/ce9974174983087641d3f814eab139d451f9101f) | reachable, tree `20c662c` | [CPU references: success](https://github.com/BLANK-2340/cuda-kernel-lab/actions/runs/42030944082) |
| [Pendulum wave simulation](https://github.com/BLANK-2340/Pendulum-Wave-simulation/blob/main/docs/SHOWCASE.md) | [`24be3e3`](https://github.com/BLANK-2340/Pendulum-Wave-simulation/commit/24be3e3b53ff69435bae32c1e4563ab7f8ebb909) | reachable, tree `e23bb39` | [Headless physics: success](https://github.com/BLANK-2340/Pendulum-Wave-simulation/actions/runs/40885566434) |

All four repositories were public with `main` as their default branch, and each
current `docs/SHOWCASE.md` path was reachable during the audit. The canonical
[`evidence_health.json`](evidence_health.json) also records the exact evidence
tree, current showcase blob, workflow identity, head revision and completion
time returned by GitHub.

Run the offline structural checks with:

```sh
python3 scripts/validate_evidence_health.py
python3 scripts/validate_evidence_health.py --self-test
```

The validator intentionally performs no network requests. It proves that the
recorded snapshot is canonical and cross-consistent with `portfolio.json`; it
does not prove that links remain reachable after November 6, 2026. Refreshing
the snapshot requires a new read-only GitHub audit and a reviewed commit.
