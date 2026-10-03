# Scalar autodiff foundation

This original educational C++17 implementation computes reverse-mode derivatives
of scalar expression graphs. No external libraries or downloaded data are needed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/scalar_demo
```

Without CMake, use GCC directly from the repository root:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I include tests/scalar_test.cpp -o scalar_test
./scalar_test
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I include examples/scalar_demo.cpp -o scalar_demo
./scalar_demo
```

Validation on October 1, 2026: GCC 13.3 compiled both executables with the warning
flags above; tests and the demo passed. The test also passed AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection disabled because this execution
environment blocks LeakSanitizer process inspection. CMake/CTest execution was
not validated here because CMake is unavailable.

Expected demo output:

```text
f = x*x + x*y
f = 15
df/dx = 8
df/dy = 3
```

## Architecture and contract

An append-only `Tape` owns nodes. `Scalar` is a handle into its tape; the tape must
outlive all handles. Tapes cannot be copied or moved. Each operation stores the
forward value and local derivatives. Backward traverses nodes in reverse creation
order, accumulates every edge (including repeated operands), and resets all
gradients before each call. The optional seed computes a scaled derivative.

Supported operations: addition, subtraction, multiplication, division, negation,
tanh, exponential, logarithm and ReLU. Division by zero, nonpositive logarithm
inputs and cross-tape operations raise exceptions. ReLU uses derivative zero at
zero. Scalars are immutable: create a new tape after updating training parameters.

Tests compare a composite expression against central finite differences, check
shared paths and repeated backward, custom seeds, disconnected nodes, ReLU
boundaries, invalid domains, tape isolation and a 100000-operation chain.

## Limitations and next milestone

CPU scalar arithmetic only; no tensors, GPU or performance claims.
Floating-point overflow/underflow follows standard double arithmetic. A tape stores
all forward nodes until destruction and is intended for bounded graphs.

The deterministic MLP, SGD and XOR extension is now available. See
[MLP_AND_XOR.md](MLP_AND_XOR.md) for its architecture, checks and actual results,
and [PORTFOLIO_PROGRESS.md](PORTFOLIO_PROGRESS.md) for the next milestone.
The existing root README is preserved; this document is the entry point for the
new implementation.
