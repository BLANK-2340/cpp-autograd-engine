# Small tensor autodiff

`include/autograd/tensor.hpp` adds a dependency-free dense tensor layer over the
existing scalar reverse-mode tape. It is intentionally small: the goal is to make
shape rules and gradient flow inspectable rather than compete with production
tensor libraries.

## Supported operations

- Dense row-major tensors with rank 1–8 and checked resource limits.
- Same-shape elementwise addition and multiplication.
- Elementwise tanh and ReLU.
- Rank-2 matrix multiplication, including zero inner dimensions.
- Sum and non-empty mean reductions.
- Numeric value and leaf-gradient extraction for tests and optimizers.

Every tensor element is an existing `Scalar` handle. Operations append ordinary
scalar nodes to one `Tape`, so tensor gradients inherit the scalar engine's
reverse traversal, shared-path accumulation and backward-reset behavior. There
is no second gradient engine.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/tensor_demo
```

The tensor test checks a known 2×3 by 3×2 product, empty inner dimensions,
elementwise repeated operands, validation paths and all 16 input/weight/bias
gradients of a `mean(tanh(XW + B)^2)` graph against central finite differences.

## Contract and limitations

Tensors are non-owning graph views: their `Tape` must outlive them. Operations do
not broadcast, mutate in place, build batched matrix products or support strides.
Matrix multiplication expands into scalar nodes, favoring clarity over speed.
Shapes are limited to eight dimensions and one million elements; these limits do
not bound total tape nodes created by a long expression. CPU `double` arithmetic
only; no GPU or tensor-performance result is claimed.

This is an original educational implementation. No third-party code or assets
were incorporated.
