# Synthetic multiclass training

This dependency-free C++17 example trains a linear three-class classifier using
the tensor layer and scalar reverse-mode tape. It is an original educational
example, not a dataset benchmark or a reproduction of a research paper.

## Run and check

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I include examples/multiclass_demo.cpp -o /tmp/autograd-multiclass-demo
/tmp/autograd-multiclass-demo
bash scripts/check.sh
```

Or build with CMake and run `./build/multiclass_demo` and CTest. The direct
check script and CTest both run the convergence test. The executable returns a
nonzero exit status if the expected default convergence is not achieved.

## Architecture and lifecycle

- Input: twelve fixed 2D training points in three separated clusters, four per
  class. Six distinct nearby points are used only after fitting as a synthetic
  holdout. No downloaded data, randomness or third-party assets are needed.
- Parameters: a row-major 2x3 weight matrix and three shared biases, all
  initialized to zero. A linear classifier can learn from zero initialization;
  this is not advice to zero-initialize hidden layers in a neural network.
- Forward: `add_rowwise(matmul(input, weights), bias)` produces 12x3 logits.
  Stable mean sparse cross-entropy supplies the training objective.
- Backward: one tape per full batch accumulates every sample's contribution
  into the same weight and bias leaves. The numeric gradients are extracted
  before the tape is destroyed.
- Update: `autograd::sgd_step` computes and validates the entire numeric
  candidate before swapping it into the parameter vector. A malformed,
  nonfinite or overflowing update cannot partially change the parameters.
- Schedule: 300 full-batch steps at learning rate 0.2, with no tuning against
  the holdout. Parameters are rebound to a fresh tape at every step.

`examples/multiclass_training.hpp` exposes the fixed experiment to the demo and
test so they execute the same training path. Tests check exact same-build
repeatability, non-overlapping points, the initial `log(3)` loss, monotone default
training loss, final accuracy, finite probabilities, and rejected/aliased SGD
updates. Primitive gradients are independently checked by the existing tensor,
classification-loss and row-broadcasting finite-difference suites.

## Measured local result — October 16, 2026

GCC 13.3, C++17, `-O2`, CPU double arithmetic:

| Metric | Result |
| --- | ---: |
| Initial training cross-entropy | 1.098612289 |
| Final training cross-entropy | 0.009116757 |
| Training accuracy | 12/12 |
| Synthetic holdout cross-entropy | 0.003424301 |
| Synthetic holdout accuracy | 6/6 |

The points are deliberately easy and linearly separable. The tiny hand-chosen
holdout does not estimate real-world generalization. Exact bitwise reproducibility
is checked within one build, not promised across different compilers or math
libraries. There is no throughput, GPU, external-data or competitive accuracy
claim. The example has fixed dimensions and is not a general classifier API.

The trained numeric weights and biases can be stored with the distinct versioned
checkpoint API documented in [CLASSIFIER_IO.md](CLASSIFIER_IO.md).
