# C++ autograd engine showcase

This dependency-free C++17 portfolio project progresses from scalar reverse-mode
automatic differentiation to a deterministic MLP and a small dense-tensor
classification example. The original root README is preserved; this page is the
concise entry point for the added implementation.

## Verify from a clean archive

Requirements: Git, Bash, `tar`, and a C++17 compiler (`g++` by default).

```sh
git clone https://github.com/BLANK-2340/cpp-autograd-engine.git
cd cpp-autograd-engine
bash scripts/verify_clean_checkout.sh
```

The script exports committed files to a temporary directory with `git archive`
and runs the strict direct-build suite there. This catches accidental dependence
on untracked files or an existing build directory. Set `CXX` to select another
compiler. The regular in-tree command is `bash scripts/check.sh`.

## Implemented milestones

| Area | Implementation | Main runnable target | Independent checks |
| --- | --- | --- | --- |
| Scalar autodiff | Append-only tape, iterative reverse traversal, nine operations | `scalar_demo` | Finite differences, shared/repeated paths, 100,000-node chain |
| MLP and SGD | Deterministic tanh MLP, shared batch bindings, atomic updates | `xor_demo` | All-parameter finite differences, three-seed convergence |
| MLP persistence | Versioned architecture and exact numeric parameter format | `model_io_demo` | Canonical round trip and malformed-input rejection |
| Dense tensors | Shapes, elementwise operations, reductions and rank-2 matmul | `tensor_demo` | 16 finite-difference gradient checks and validation |
| Classification | Stable row softmax, sparse cross-entropy and row bias | `classification_demo` | Extreme logits and 15 combined finite-difference checks |
| Multiclass training | Fresh-tape full-batch linear classifier | `multiclass_demo` | Repeatability, holdout, atomic SGD and convergence |
| Classifier checkpoint | Versioned weights/biases plus numeric continuation | `linear_classifier_io_demo` | Exact values, malformed input and 120+180 resume equivalence |

## Actually observed results

These results come from the fixed educational examples; they are not external
dataset or performance claims.

| Example | Result |
| --- | ---: |
| Scalar `f=x*x+x*y` at `x=3, y=2` | `f=15`, `df/dx=8`, `df/dy=3` |
| XOR seed 42 after 4,000 updates | MSE `0.000218520`, accuracy `4/4` |
| Synthetic multiclass training | Cross-entropy `1.098612289` → `0.009116757`, accuracy `12/12` |
| Disjoint synthetic holdout | Cross-entropy `0.003424301`, accuracy `6/6` |
| Checkpointed continuation | 120+180 steps exactly equal 300 uninterrupted steps in the same build |

The strict suite currently builds and executes eight tests and seven demos with
`-Wall -Wextra -Werror -pedantic`. Architecture details and narrower limitations
are documented in [GETTING_STARTED.md](GETTING_STARTED.md),
[MLP_AND_XOR.md](MLP_AND_XOR.md), [MODEL_IO.md](MODEL_IO.md),
[TENSORS.md](TENSORS.md), [MULTICLASS.md](MULTICLASS.md), and
[CLASSIFIER_IO.md](CLASSIFIER_IO.md).

## Known limitations

- CPU `double` arithmetic only; the tape retains a complete bounded graph until
  destruction and tensors are educational scalar views rather than optimized
  storage kernels.
- Exact repeatability is checked within one build. Results are not promised to be
  bitwise identical across compilers, standard libraries or architectures.
- The synthetic classifier is deliberately easy and does not establish
  real-world generalization. No throughput, GPU or competitive-accuracy claim is
  made.
- Persistence formats exclude checksums, authentication, atomic filesystem
  replacement and stateful-optimizer state.

## Milestone commits

- [Scalar engine and gradient checks](https://github.com/BLANK-2340/cpp-autograd-engine/commit/3ca9993f3d8cd5136b6b9d191b842ce248c3f5ec)
- [Deterministic MLP and XOR](https://github.com/BLANK-2340/cpp-autograd-engine/commit/3b422dcd893ddfd1d8eddf91814c0129b5f22012)
- [MLP serialization](https://github.com/BLANK-2340/cpp-autograd-engine/commit/3ed0a908aeb1fd15b7f263d0be39bdda4a001761)
- [Tensor primitives](https://github.com/BLANK-2340/cpp-autograd-engine/commit/1d87ea170a57fb6ff7ce6bfcc4655af105aff7c2)
- [Stable classification loss](https://github.com/BLANK-2340/cpp-autograd-engine/commit/550417784d61982fc054c16f590a229a5ce1f995)
- [Row broadcasting](https://github.com/BLANK-2340/cpp-autograd-engine/commit/90b4c001a336cc9ffc78fdac658ef57689b0c81a)
- [Multiclass training](https://github.com/BLANK-2340/cpp-autograd-engine/commit/75f0673e7998598c5b014d92157a297fe57124b6)
- [Classifier checkpoints](https://github.com/BLANK-2340/cpp-autograd-engine/commit/7193a528907ed735bdb4030bc3970eb48c164da1)
- [Exact resumed training](https://github.com/BLANK-2340/cpp-autograd-engine/commit/eced76e06f10be5c38763529b6e24c61505fb764)
