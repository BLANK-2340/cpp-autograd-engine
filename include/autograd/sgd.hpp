#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace autograd {

// Numeric parameters only: tape handles are rebound after every update. All
// candidates are checked before mutation, including when gradients alias values.
inline void sgd_step(std::vector<double>& parameters,
                     const std::vector<double>& gradients,
                     double learning_rate) {
    if (parameters.size() != gradients.size())
        throw std::invalid_argument("SGD parameter and gradient counts differ");
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("SGD learning rate must be positive and finite");
    auto candidate = parameters;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (!std::isfinite(parameters[index]) || !std::isfinite(gradients[index]))
            throw std::invalid_argument("SGD values and gradients must be finite");
        candidate[index] -= learning_rate * gradients[index];
        if (!std::isfinite(candidate[index]))
            throw std::overflow_error("SGD candidate is not finite");
    }
    parameters.swap(candidate);
}

} // namespace autograd
