#pragma once

#include "autograd/scalar.hpp"
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

namespace autograd {

// One immutable parameter snapshot bound to one tape. Reuse it for every sample
// in a batch so backward accumulates all sample contributions into the same leaves.
class BoundMLP {
    Tape* tape_;
    std::vector<std::size_t> widths_;
    std::vector<Scalar> parameters_;
public:
    BoundMLP(Tape& tape, std::vector<std::size_t> widths,
             const std::vector<double>& parameters)
        : tape_(&tape), widths_(std::move(widths)) {
        std::size_t expected = 0;
        if (widths_.size() < 2) throw std::invalid_argument("MLP needs at least two widths");
        for (auto width : widths_)
            if (width == 0) throw std::invalid_argument("MLP widths must be positive");
        for (std::size_t i = 1; i < widths_.size(); ++i) {
            const auto in = widths_[i - 1], out = widths_[i];
            const auto limit = std::numeric_limits<std::size_t>::max();
            if (in == limit || out > (limit - expected) / (in + 1))
                throw std::invalid_argument("MLP parameter count overflows");
            expected += out * (in + 1);
        }
        if (parameters.size() != expected)
            throw std::invalid_argument("MLP parameter count mismatch");
        for (double p : parameters)
            if (!std::isfinite(p)) throw std::invalid_argument("Parameters must be finite");
        parameters_.reserve(expected);
        for (double p : parameters) parameters_.push_back(tape.scalar(p));
    }

    const std::vector<Scalar>& parameters() const { return parameters_; }

    // All layers use tanh; outputs lie in [-1, 1]. The tape must outlive this object.
    std::vector<Scalar> operator()(const std::vector<Scalar>& input) const {
        if (input.size() != widths_.front())
            throw std::invalid_argument("MLP input width mismatch");
        for (auto x : input) tape_->require(x);
        auto activation = input;
        std::size_t offset = 0;
        for (std::size_t layer = 1; layer < widths_.size(); ++layer) {
            std::vector<Scalar> next;
            next.reserve(widths_[layer]);
            for (std::size_t neuron = 0; neuron < widths_[layer]; ++neuron) {
                auto sum = parameters_[offset + activation.size()];
                for (std::size_t j = 0; j < activation.size(); ++j)
                    sum = sum + parameters_[offset + j] * activation[j];
                offset += activation.size() + 1;
                next.push_back(tanh(sum));
            }
            activation = std::move(next);
        }
        return activation;
    }

    std::vector<double> gradients() const {
        std::vector<double> result;
        result.reserve(parameters_.size());
        for (auto p : parameters_) result.push_back(p.grad());
        return result;
    }
};

// Owns numeric parameters, never tape handles. Flat order is layer, neuron,
// incoming weights, bias. Bind once per batch; destroy the tape after the update.
class MLP {
    std::vector<std::size_t> widths_;
    std::vector<double> parameters_;
public:
    explicit MLP(std::vector<std::size_t> widths, std::uint32_t seed = 42)
        : widths_(std::move(widths)) {
        if (widths_.size() < 2) throw std::invalid_argument("MLP needs at least two widths");
        for (auto width : widths_)
            if (width == 0) throw std::invalid_argument("MLP widths must be positive");
        std::size_t count = 0;
        for (std::size_t i = 1; i < widths_.size(); ++i) {
            const auto in = widths_[i - 1], out = widths_[i];
            const auto limit = std::numeric_limits<std::size_t>::max();
            if (in == limit || out > (limit - count) / (in + 1))
                throw std::invalid_argument("MLP parameter count overflows");
            count += out * (in + 1);
        }
        parameters_.reserve(count);
        std::mt19937 rng(seed);
        for (std::size_t layer = 1; layer < widths_.size(); ++layer) {
            const auto in = widths_[layer - 1], out = widths_[layer];
            const double scale = std::sqrt(6.0 / (static_cast<double>(in) + out));
            for (std::size_t neuron = 0; neuron < out; ++neuron) {
                for (std::size_t j = 0; j < in; ++j) {
                    // Specify the mapping instead of implementation-dependent distributions.
                    const double u = static_cast<double>(rng()) / 4294967296.0;
                    parameters_.push_back((2.0 * u - 1.0) * scale);
                }
                parameters_.push_back(0.0);
            }
        }
    }

    const std::vector<double>& parameters() const { return parameters_; }
    const std::vector<std::size_t>& widths() const { return widths_; }
    BoundMLP bind(Tape& tape) const { return BoundMLP(tape, widths_, parameters_); }

    void set_parameters(const std::vector<double>& values) {
        if (values.size() != parameters_.size())
            throw std::invalid_argument("MLP parameter count mismatch");
        for (double value : values)
            if (!std::isfinite(value)) throw std::invalid_argument("Parameters must be finite");
        parameters_ = values;
    }

    // Compute the complete candidate first: invalid updates cannot partially mutate the model.
    void sgd(const std::vector<double>& gradients, double learning_rate) {
        if (gradients.size() != parameters_.size())
            throw std::invalid_argument("MLP gradient count mismatch");
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
            throw std::invalid_argument("Learning rate must be positive and finite");
        auto updated = parameters_;
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            if (!std::isfinite(gradients[i]))
                throw std::invalid_argument("Gradients must be finite");
            updated[i] -= learning_rate * gradients[i];
            if (!std::isfinite(updated[i])) throw std::overflow_error("SGD update is nonfinite");
        }
        parameters_.swap(updated);
    }
};
} // namespace autograd
