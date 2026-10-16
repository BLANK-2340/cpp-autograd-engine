#include "autograd/tensor.hpp"

#include <cmath>
#include <iostream>
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

template <typename Exception = std::invalid_argument, typename Function>
void rejects(Function function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, "Expected validation exception");
}

double objective(const std::vector<double>& matrix,
                 const std::vector<double>& bias) {
    double total = 0.0;
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        const double activation = std::tanh(matrix[index] + bias[index % 3]);
        total += activation * activation / static_cast<double>(matrix.size());
    }
    return total;
}

void forward_and_gradient_checks() {
    using namespace autograd;
    const std::vector<double> matrix_values{0.2, -0.4, 0.7, 0.5, 0.1, -0.3};
    const std::vector<double> bias_values{0.1, -0.2, 0.3};
    Tape tape;
    Tensor matrix(tape, {2, 3}, matrix_values);
    Tensor bias(tape, {3}, bias_values);

    const Tensor shifted = add_rowwise(matrix, bias);
    require(shifted.shape() == std::vector<std::size_t>({2, 3}),
            "Row-wise output shape mismatch");
    const std::vector<double> expected{0.3, -0.6, 1.0, 0.6, -0.1, 0.0};
    for (std::size_t index = 0; index < expected.size(); ++index)
        near(shifted[index].value(), expected[index], "Row-wise forward mismatch", 1e-12);

    const Tensor activation = tanh(shifted);
    const Scalar loss = mean(activation * activation);
    near(loss.value(), objective(matrix_values, bias_values),
         "Row-wise objective mismatch", 1e-12);
    tape.backward(loss);

    const double step = 1e-5;
    for (std::size_t index = 0; index < matrix_values.size(); ++index) {
        auto plus = matrix_values;
        auto minus = matrix_values;
        plus[index] += step;
        minus[index] -= step;
        const double numeric = (objective(plus, bias_values) -
                                objective(minus, bias_values)) /
                               (2.0 * step);
        near(matrix.gradients()[index], numeric,
             "Matrix finite-difference gradient mismatch");
    }
    for (std::size_t index = 0; index < bias_values.size(); ++index) {
        auto plus = bias_values;
        auto minus = bias_values;
        plus[index] += step;
        minus[index] -= step;
        const double numeric = (objective(matrix_values, plus) -
                                objective(matrix_values, minus)) /
                               (2.0 * step);
        near(bias.gradients()[index], numeric,
             "Broadcast bias finite-difference gradient mismatch");

        const double accumulated = matrix.gradients()[index] +
                                   matrix.gradients()[index + 3];
        near(bias.gradients()[index], accumulated,
             "Broadcast bias did not accumulate every row", 1e-12);
    }

    const auto first_bias_gradient = bias.gradients();
    tape.backward(loss);
    require(bias.gradients() == first_bias_gradient,
            "Repeated backward did not reset broadcast gradients");
}

void boundary_and_validation_checks() {
    using namespace autograd;
    Tape tape, other;
    Tensor empty_rows(tape, {0, 3}, {});
    Tensor row(tape, {3}, {1.0, 2.0, 3.0});
    require(add_rowwise(empty_rows, row).empty(),
            "Zero-row broadcast output is not empty");

    Tensor empty_columns(tape, {2, 0}, {});
    Tensor empty_row(tape, {0}, {});
    require(add_rowwise(empty_columns, empty_row).empty(),
            "Zero-column broadcast output is not empty");

    Tensor matrix(tape, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor wrong_width(tape, {2}, {1, 2});
    Tensor rank_two_row(tape, {1, 3}, {1, 2, 3});
    Tensor foreign(other, {3}, {1, 2, 3});
    rejects([&] { (void)add_rowwise(row, row); });
    rejects([&] { (void)add_rowwise(matrix, wrong_width); });
    rejects([&] { (void)add_rowwise(matrix, rank_two_row); });
    rejects([&] { (void)add_rowwise(matrix, foreign); });
}
} // namespace

int main() {
    forward_and_gradient_checks();
    boundary_and_validation_checks();
    std::cout << "PASS: explicit row broadcasting, boundary validation, and nine "
                 "finite-difference gradients\n";
}
