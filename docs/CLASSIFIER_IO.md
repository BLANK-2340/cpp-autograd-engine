# Versioned linear-classifier checkpoints

`autograd/linear_classifier_io.hpp` persists a dense multiclass linear model as
numeric dimensions, row-major weights and one bias per class. It is separate
from the established MLP format and does not serialize tape nodes, gradients,
optimizer state or executable code.

```cpp
autograd::LinearClassifierModel model{2, 3, trained_parameters};
const std::string checkpoint = autograd::serialize_linear_classifier(model);
const auto restored = autograd::deserialize_linear_classifier(checkpoint);
```

The stream APIs are `save_linear_classifier` and `load_linear_classifier`.
`linear_classifier_io_demo` trains the synthetic classifier, round-trips it in
memory, resumes from the restored numeric parameters and verifies the result.

## Deterministic resume contract

`multiclass_demo::continue_training(parameters, steps, learning_rate)` starts a
new sequence of fresh per-batch tapes from validated finite numeric parameters.
For the fixed full-batch experiment, 120 steps, an exact checkpoint round trip
and another 180 steps produce the same parameters and loss trajectory as 300
uninterrupted steps in the same build. The test compares both exactly.

The checkpoint does not claim to capture an optimizer: the example uses plain
stateless SGD with an explicit learning rate. A future optimizer with momentum
or adaptive statistics would need a separately versioned training-state format.

## Format and validation

Version 1 is a locale-independent whitespace-delimited format:

```text
AUTOGRAD_LINEAR_CLASSIFIER 1
input_features <positive count>
classes <count of at least two>
parameters <(input_features + 1) * classes>
<finite row-major weights, then biases>
end
```

Finite doubles are emitted with `max_digits10`, preserving their exact values
on a conforming C++17 implementation. Loading rejects unknown versions, invalid
dimensions, count mismatches, nonfinite or truncated parameters, trailing data
and models above 10,000,000 parameters before allocating the parameter vector.

This educational checkpoint is not a general model-interchange standard. It has
no checksum, authentication, optimizer state or atomic file-replacement helper;
callers remain responsible for trusted storage and safe file replacement.
