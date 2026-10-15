#include "autograd/tensor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const std::string& message,
          double tolerance = 2e-6) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
            message);
}

template <typename Exception, typename Function>
void rejects(Function function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, "Expected validation exception");
}

double numeric_cross_entropy(const std::vector<double>& logits,
                             const std::vector<std::size_t>& targets) {
    const std::size_t columns = logits.size() / targets.size();
    double total = 0.0;
    for (std::size_t row = 0; row < targets.size(); ++row) {
        double maximum = logits[row * columns];
        for (std::size_t column = 1; column < columns; ++column)
            maximum = std::fmax(maximum, logits[row * columns + column]);
        double denominator = 0.0;
        for (std::size_t column = 0; column < columns; ++column)
            denominator += std::exp(logits[row * columns + column] - maximum);
        total += std::log(denominator) + maximum -
                 logits[row * columns + targets[row]];
    }
    return total / static_cast<double>(targets.size());
}

void probability_and_gradient_checks() {
    using namespace autograd;
    const std::vector<double> values{0.3, -1.2, 2.1, -0.7, 0.5, 1.4};
    const std::vector<std::size_t> targets{2, 0};
    Tape tape;
    Tensor logits(tape, {2, 3}, values);
    const Tensor probabilities = softmax_rows(logits);

    for (std::size_t row = 0; row < 2; ++row) {
        double row_sum = 0.0;
        for (std::size_t column = 0; column < 3; ++column) {
            const double probability = probabilities.at(row, column).value();
            require(probability > 0.0 && probability < 1.0,
                    "Softmax probability outside open unit interval");
            row_sum += probability;
        }
        near(row_sum, 1.0, "Softmax row does not sum to one", 1e-12);
    }

    const Scalar loss = cross_entropy(logits, targets);
    near(loss.value(), numeric_cross_entropy(values, targets),
         "Cross-entropy forward mismatch", 1e-12);
    tape.backward(loss);

    const double step = 1e-5;
    for (std::size_t index = 0; index < values.size(); ++index) {
        auto plus = values;
        auto minus = values;
        plus[index] += step;
        minus[index] -= step;
        const double numeric = (numeric_cross_entropy(plus, targets) -
                                numeric_cross_entropy(minus, targets)) /
                               (2.0 * step);
        near(logits.gradients()[index], numeric,
             "Cross-entropy finite-difference gradient mismatch");

        const std::size_t row = index / 3;
        const std::size_t column = index % 3;
        const double expected = (probabilities[index].value() -
                                 (column == targets[row] ? 1.0 : 0.0)) /
                                2.0;
        near(logits.gradients()[index], expected,
             "Cross-entropy analytic gradient mismatch", 1e-12);
    }
}

void stability_checks() {
    using namespace autograd;
    Tape tape;
    Tensor logits(tape, {2, 3}, {10000.0, 10001.0, 9999.0,
                                 -10000.0, -10002.0, -9998.0});
    const Tensor probabilities = softmax_rows(logits);
    const Scalar loss = cross_entropy(logits, {1, 2});
    require(std::isfinite(loss.value()), "Extreme-logit loss is not finite");
    tape.backward(loss);
    for (const double value : probabilities.values())
        require(std::isfinite(value), "Extreme-logit probability is not finite");
    for (const double gradient : logits.gradients())
        require(std::isfinite(gradient), "Extreme-logit gradient is not finite");

    Tape shifted_tape;
    Tensor shifted(shifted_tape, {2, 3}, {22345.0, 22346.0, 22344.0,
                                          2345.0, 2343.0, 2347.0});
    near(cross_entropy(shifted, {1, 2}).value(), loss.value(),
         "Cross-entropy is not invariant to a shared shift", 1e-12);
}

void validation_checks() {
    using namespace autograd;
    Tape tape;
    Tensor vector(tape, {3}, {1.0, 2.0, 3.0});
    rejects<std::invalid_argument>([&] { (void)softmax_rows(vector); });
    rejects<std::invalid_argument>([&] { (void)cross_entropy(vector, {0}); });

    Tensor no_rows(tape, {0, 3}, {});
    require(softmax_rows(no_rows).empty(), "Empty softmax output mismatch");
    rejects<std::domain_error>([&] { (void)cross_entropy(no_rows, {}); });
    Tensor no_classes(tape, {2, 0}, {});
    rejects<std::domain_error>([&] { (void)softmax_rows(no_classes); });

    Tensor logits(tape, {2, 2}, {1.0, 2.0, 3.0, 4.0});
    rejects<std::invalid_argument>([&] { (void)cross_entropy(logits, {0}); });
    rejects<std::out_of_range>([&] { (void)cross_entropy(logits, {0, 2}); });
    Tensor nonfinite(tape, {1, 2},
                     {0.0, std::numeric_limits<double>::infinity()});
    rejects<std::domain_error>([&] { (void)softmax_rows(nonfinite); });
    rejects<std::domain_error>([&] { (void)cross_entropy(nonfinite, {0}); });
}
} // namespace

int main() {
    probability_and_gradient_checks();
    stability_checks();
    validation_checks();
    std::cout << "PASS: stable row softmax, sparse cross-entropy, six "
                 "finite-difference gradients, and validation\n";
}
