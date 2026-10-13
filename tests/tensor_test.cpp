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

double objective(const std::vector<double>& x,
                 const std::vector<double>& weights,
                 const std::vector<double>& bias) {
    double loss = 0.0;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            double value = bias[row * 2 + column];
            for (std::size_t inner = 0; inner < 3; ++inner)
                value += x[row * 3 + inner] * weights[inner * 2 + column];
            const double activation = std::tanh(value);
            loss += activation * activation / 4.0;
        }
    }
    return loss;
}

void finite_difference_checks() {
    using namespace autograd;
    const std::vector<double> x_values{0.2, -0.4, 0.7, 0.5, 0.1, -0.3};
    const std::vector<double> weight_values{0.3, -0.2, 0.6, 0.4, -0.5, 0.8};
    const std::vector<double> bias_values{0.1, -0.1, 0.2, 0.05};

    Tape tape;
    Tensor x(tape, {2, 3}, x_values);
    Tensor weights(tape, {3, 2}, weight_values);
    Tensor bias(tape, {2, 2}, bias_values);
    const Tensor activation = tanh(matmul(x, weights) + bias);
    const Scalar loss = mean(activation * activation);
    near(loss.value(), objective(x_values, weight_values, bias_values),
         "Forward objective mismatch", 1e-12);
    tape.backward(loss);

    const double step = 1e-5;
    const auto check_group = [&](const std::vector<double>& values,
                                 const std::vector<double>& gradients,
                                 int group) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto plus_x = x_values, minus_x = x_values;
            auto plus_w = weight_values, minus_w = weight_values;
            auto plus_b = bias_values, minus_b = bias_values;
            std::vector<double>* plus[] = {&plus_x, &plus_w, &plus_b};
            std::vector<double>* minus[] = {&minus_x, &minus_w, &minus_b};
            (*plus[group])[index] += step;
            (*minus[group])[index] -= step;
            const double numeric =
                (objective(plus_x, plus_w, plus_b) -
                 objective(minus_x, minus_w, minus_b)) /
                (2.0 * step);
            near(gradients[index], numeric,
                 "Finite-difference gradient mismatch at group " +
                     std::to_string(group) + " index " + std::to_string(index));
        }
    };
    check_group(x_values, x.gradients(), 0);
    check_group(weight_values, weights.gradients(), 1);
    check_group(bias_values, bias.gradients(), 2);

    const auto first_gradients = x.gradients();
    tape.backward(loss);
    require(x.gradients() == first_gradients, "Repeated backward did not reset gradients");
}

void primitive_checks() {
    using namespace autograd;
    Tape tape;
    Tensor lhs(tape, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor rhs(tape, {3, 2}, {7, 8, 9, 10, 11, 12});
    const Tensor product = matmul(lhs, rhs);
    require(product.shape() == std::vector<std::size_t>({2, 2}),
            "Matrix output shape mismatch");
    require(product.values() == std::vector<double>({58, 64, 139, 154}),
            "Known matrix product mismatch");
    require(product.at(1, 0).value() == 139, "Matrix indexing mismatch");

    Tensor values(tape, {2}, {-2.0, 3.0});
    require(relu(values).values() == std::vector<double>({0.0, 3.0}),
            "Tensor ReLU mismatch");
    const Tensor repeated = values * values + values;
    const Scalar total = sum(repeated);
    tape.backward(total);
    require(values.gradients() == std::vector<double>({-3.0, 7.0}),
            "Elementwise repeated-operand gradient mismatch");

    Tensor empty_lhs(tape, {2, 0}, {});
    Tensor empty_rhs(tape, {0, 3}, {});
    require(matmul(empty_lhs, empty_rhs).values() == std::vector<double>(6, 0.0),
            "Zero-inner-dimension product mismatch");
}

void validation_checks() {
    using namespace autograd;
    Tape tape, other;
    rejects([&] { Tensor invalid(tape, {}, {}); });
    rejects([&] { Tensor invalid(tape, {2, 2}, {1.0}); });
    rejects([&] {
        Tensor invalid(tape, {1000000, 2}, std::vector<double>{});
    });
    rejects([&] {
        Tensor invalid(tape, {1000001}, std::vector<double>{});
    });

    Tensor a(tape, {2}, {1, 2});
    Tensor b(tape, {1, 2}, {1, 2});
    Tensor c(other, {2}, {1, 2});
    rejects([&] { (void)(a + b); });
    rejects([&] { (void)(a + c); });
    rejects([&] { (void)matmul(a, b); });
    rejects<std::out_of_range>([&] { (void)b.at(1, 0); });

    Tensor matrix(tape, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor wrong_inner(tape, {2, 2}, {1, 2, 3, 4});
    rejects([&] { (void)matmul(matrix, wrong_inner); });
    Tensor empty(tape, {0}, {});
    rejects<std::domain_error>([&] { (void)mean(empty); });
}
} // namespace

int main() {
    finite_difference_checks();
    primitive_checks();
    validation_checks();
    std::cout << "PASS: tensor elementwise ops, reductions, matmul, validation, "
                 "and 16 finite-difference gradients\n";
}
