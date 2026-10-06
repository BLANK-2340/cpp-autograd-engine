# Versioned MLP model format

The header-only `autograd/model_io.hpp` adds dependency-free persistence for the
numeric `MLP`. It serializes architecture plus parameters—not tape nodes,
gradients, optimizer state or executable code.

## API and demo

```cpp
#include "autograd/model_io.hpp"
#include <fstream>

autograd::MLP model({2, 4, 1}, 42);
std::ofstream output("model.agmlp");
autograd::save_model(output, model);
output.close();

std::ifstream input("model.agmlp");
autograd::MLP restored = autograd::load_model(input);
```

String helpers `serialize_model` and `deserialize_model` are also available.
Run `./build/model_io_demo`, or use `bash scripts/check.sh` to build and run
the complete suite.

## Format contract

Version 1 is a whitespace-delimited, locale-independent text format:

```text
AUTOGRAD_MLP 1
widths <count> <width...>
parameters <count>
<one finite double per parameter>
end
```

Parameters retain the MLP's flat layer/neuron/weights/bias order. Saving uses the
classic locale, scientific notation and `std::numeric_limits<double>::max_digits10`;
this is sufficient for an exact finite-double round trip on conforming C++17
implementations. The human-readable format is deterministic for the same model.

The loader rejects unknown versions, invalid or excessive layer counts, zero
widths, architectures above 10,000,000 parameters, mismatched counts, nonfinite
parameters, truncated input and trailing tokens. It validates the whole document
before returning a model. This cap prevents a small untrusted document from
requesting unbounded allocation; the format is not a hardened sandbox for
hostile inputs.

## Scope and limitations

This is a reproducibility checkpoint for the educational CPU MLP. It is not a
general interchange standard and has no backward-compatibility promise beyond
explicitly supported versions. It does not preserve optimizer state, tape graphs,
gradients, metadata or checksums. Callers remain responsible for secure file
permissions, storage integrity and atomic filesystem replacement when saving to
disk.

