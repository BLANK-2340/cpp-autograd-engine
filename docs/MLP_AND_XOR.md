# Deterministic MLP and XOR

This original, dependency-free C++17 extension uses the existing scalar tape to
train a fully connected network. It adds no third-party code or datasets.

## Build and run

From the repository root, compile and run all tests and both demos:

```sh
bash scripts/check.sh
```

Or build just XOR:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I include examples/xor_demo.cpp -o /tmp/xor_demo
/tmp/xor_demo
```

With CMake installed:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/xor_demo
```

## Parameter and tape contract

`MLP({2, 4, 1}, 42)` owns 17 numeric parameters. Flat ordering is layer, neuron,
incoming weights, then bias. Each layer, including the output, uses tanh. Weights
start uniformly within `sqrt(6 / (fan_in + fan_out))` of zero; biases start at zero.
The initialization maps raw `std::mt19937` values to doubles explicitly, avoiding
library-specific `uniform_real_distribution` behavior. Floating-point arithmetic
and math libraries can still produce small differences between platforms.

`model.bind(tape)` copies the current parameters into leaves on that tape and
returns a `BoundMLP`. Reuse the same binding across the full batch. Calling
`tape.backward(loss)` accumulates contributions from every sample into those leaves
and resets previous gradients. `bound.gradients()` returns a numeric vector, and
`model.sgd(gradients, learning_rate)` updates the model. The learning rate must be
positive and finite; the complete update must remain finite.

Every iteration constructs a fresh tape and binding. Existing bindings keep their
forward-time parameter snapshot after an update. The tape must outlive bindings
and all scalar handles; the model itself need not outlive a binding. Do not reuse
stale gradients after changing parameters. Vector gradients carry no model identity:
callers must pair gradients with the model that produced them.

Architecture and input dimensions are validated, including size overflow. Parameter
replacement and SGD reject invalid sizes and nonfinite values. Failed validation or
overflow leaves all numeric parameters unchanged. Inputs follow the scalar engine's
ordinary double arithmetic, so nonfinite inputs can propagate nonfinite outputs.

## Actual results on October 3, 2026

GCC 13.3 on this execution environment, strict warning flags and `-O2`, CPU only.
XOR uses all four truth-table rows as a training batch. Inputs and targets are
encoded as -1 and +1. Classification uses the output sign. Full-batch SGD uses
learning rate 0.1 for exactly 4000 updates, without tuning during the test.

| Seed | Final mean squared error | Correct truth-table outputs |
| --- | ---: | ---: |
| 7 | 0.000374997 | 4/4 |
| 42 | 0.000218520 | 4/4 |
| 2026 | 0.000388960 | 4/4 |

Actual seed-42 demo output:

```text
XOR: 2-4-1 tanh MLP, seed=42, full-batch SGD, lr=0.1
step=0 mse=1.443258
step=1000 mse=0.001038
step=2000 mse=0.000469
step=3000 mse=0.000299
step=4000 mse=0.000219
(-1.000000, -1.000000) target=-1.000000 prediction=-0.984238
(-1.000000, 1.000000) target=1.000000 prediction=0.984774
(1.000000, -1.000000) target=1.000000 prediction=0.985413
(1.000000, 1.000000) target=-1.000000 prediction=-0.986546
accuracy=4/4
```

This establishes toy optimization correctness on the complete XOR truth table,
not generalization to an unseen dataset or a performance benchmark. No GPU results
are claimed.

## Verification and limitations

Tests use an independent double-only forward implementation and central finite
differences for every parameter of a two-sample, multi-output network. They also
check a hand-computed neuron, input derivatives, repeated backward, exact SGD
updates, snapshot behavior, initialization, bad dimensions, cross-tape inputs,
nonfinite gradients and atomic failure handling. XOR must reach MSE below 0.01 and
4/4 correct outputs for each registered seed. Original scalar checks still run.

CPU scalar graphs only; no tensors, minibatch loader, optimizer state or model
serialization. Memory grows with a batch's graph until its tape is destroyed.
The API is educational and does not offer production training throughput.
