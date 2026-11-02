# Portfolio release checklist

This deterministic checklist answers one release question: does every public
showcase expose setup guidance, a runnable demo, an automated check, a pinned
evidence revision and an adjacent limitation? It does not rerun the other
repositories or replace their own verification scripts.

| Project | Setup | Demo | Checks | Evidence | Limitations |
| --- | --- | --- | --- | --- | --- |
| C++ autograd engine | [Showcase guide](https://github.com/BLANK-2340/cpp-autograd-engine/blob/main/docs/SHOWCASE.md) | [Runnable examples](https://github.com/BLANK-2340/cpp-autograd-engine/blob/main/docs/SHOWCASE.md) | `bash scripts/verify_clean_checkout.sh` | [Pinned revision](https://github.com/BLANK-2340/cpp-autograd-engine/commit/32e5cdf5a50f91ce12bd187a15b9ab4660cdedf0) | Scope: dependency-free educational CPU double arithmetic. The synthetic data does not establish real-world generalization, throughput or competitive accuracy. |
| VisionTrack-NN | [Showcase guide](https://github.com/BLANK-2340/VisionTrack-NN/blob/main/docs/SHOWCASE.md) | [Runnable examples](https://github.com/BLANK-2340/VisionTrack-NN/blob/main/docs/SHOWCASE.md) | `./scripts/verify_showcase.sh` | [Pinned revision](https://github.com/BLANK-2340/VisionTrack-NN/commit/12674754043ee6e7b097a549078736b5f913be01) | Scope: Fashion-MNIST figures use fixed balanced subsets, not the full official train/test sets. |
| CUDA kernel lab | [Showcase guide](https://github.com/BLANK-2340/cuda-kernel-lab/blob/main/docs/SHOWCASE.md) | [Runnable examples](https://github.com/BLANK-2340/cuda-kernel-lab/blob/main/docs/SHOWCASE.md) | `./scripts/verify_showcase.sh` | [Pinned revision](https://github.com/BLANK-2340/cuda-kernel-lab/commit/8bdf0620b85e30d6398a0a64627ce9200c62a611) | Scope: allocating and preallocated modes measure different timing boundaries; their ratios are not kernel speedups. |
| Pendulum wave simulation | [Showcase guide](https://github.com/BLANK-2340/Pendulum-Wave-simulation/blob/main/docs/SHOWCASE.md) | [Runnable examples](https://github.com/BLANK-2340/Pendulum-Wave-simulation/blob/main/docs/SHOWCASE.md) | `./scripts/verify_showcase.sh` | [Pinned revision](https://github.com/BLANK-2340/Pendulum-Wave-simulation/commit/24be3e3b53ff69435bae32c1e4563ab7f8ebb909) | Scope: the fine-step trajectory is a numerical reference, not an analytical solution. |

`python3 scripts/validate_release_checklist.py` strictly cross-validates the
canonical JSON companion against `portfolio.json` and this table. Passing means
all five release facets are present and synchronized; it does not assert that
external links remain live after the dated evidence-health audit.
