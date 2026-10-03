#include "autograd/mlp.hpp"
#include <array>
#include <iostream>
#include <string>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string& message, double tolerance = 1e-6) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)), message);
}
template <class Exception = std::invalid_argument, class F>
void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const Exception&) { caught = true; }
    require(caught, "Expected validation exception");
}

// Independent numeric forward path: no autodiff operators or parameter binding.
std::vector<double> reference(const std::vector<std::size_t>& widths,
                              const std::vector<double>& params, std::vector<double> x) {
    std::size_t offset = 0;
    for (std::size_t layer = 1; layer < widths.size(); ++layer) {
        std::vector<double> y(widths[layer]);
        for (auto& value : y) {
            value = params.at(offset + x.size());
            for (std::size_t j = 0; j < x.size(); ++j) value += params.at(offset + j) * x[j];
            offset += x.size() + 1;
            value = std::tanh(value);
        }
        x = std::move(y);
    }
    return x;
}

void gradient_checks() {
    using namespace autograd;
    MLP model({2, 3, 2}, 7);
    const std::vector<std::vector<double>> samples{{0.2, -0.7}, {0.8, 0.4}};
    const std::vector<std::vector<double>> targets{{0.4, -0.2}, {-0.1, 0.5}};
    Tape tape;
    auto bound = model.bind(tape);
    auto loss = tape.scalar(0);
    for (std::size_t sample = 0; sample < samples.size(); ++sample) {
        auto output = bound({tape.scalar(samples[sample][0]), tape.scalar(samples[sample][1])});
        const auto numeric = reference(model.widths(), model.parameters(), samples[sample]);
        for (std::size_t j = 0; j < output.size(); ++j) {
            near(output[j].value(), numeric[j], "Multi-output forward");
            auto error = output[j] - tape.scalar(targets[sample][j]);
            loss = loss + error * error;
        }
    }
    tape.backward(loss);
    const auto gradients = bound.gradients();
    auto objective = [&](const std::vector<double>& params) {
        double result = 0;
        for (std::size_t sample = 0; sample < samples.size(); ++sample) {
            const auto output = reference(model.widths(), params, samples[sample]);
            for (std::size_t j = 0; j < output.size(); ++j) {
                const double error = output[j] - targets[sample][j];
                result += error * error;
            }
        }
        return result;
    };
    const double h = 1e-5;
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        auto plus = model.parameters(), minus = plus;
        plus[i] += h; minus[i] -= h;
        near(gradients[i], (objective(plus) - objective(minus)) / (2 * h),
             "Batch parameter gradient " + std::to_string(i));
    }
    tape.backward(loss);
    require(bound.gradients() == gradients, "Batch backward reset");

    // A neuron with hand-assigned parameters also validates flat ordering and input derivatives.
    MLP neuron({2, 1});
    neuron.set_parameters({0.5, -0.3, 0.2});
    Tape single;
    auto n = neuron.bind(single);
    auto x = single.scalar(0.4), y = single.scalar(-0.6);
    auto out = n({x, y})[0];
    single.backward(out);
    const double expected = std::tanh(0.58), slope = 1 - expected * expected;
    near(out.value(), expected, "Neuron hand-computed output");
    near(x.grad(), 0.5 * slope, "Neuron input x");
    near(y.grad(), -0.3 * slope, "Neuron input y");
    near(n.parameters()[0].grad(), 0.4 * slope, "Neuron weight x");
    near(n.parameters()[1].grad(), -0.6 * slope, "Neuron weight y");
    near(n.parameters()[2].grad(), slope, "Neuron bias");
    const auto before = neuron.parameters();
    const auto g = n.gradients();
    neuron.sgd(g, 0.1);
    for (std::size_t i = 0; i < g.size(); ++i)
        near(neuron.parameters()[i], before[i] - 0.1 * g[i], "SGD update");
    near(n({x, y})[0].value(), expected, "Existing binding is an immutable snapshot");
    Tape fresh;
    auto new_binding = neuron.bind(fresh);
    near(new_binding({fresh.scalar(0.4), fresh.scalar(-0.6)})[0].value(),
         reference(neuron.widths(), neuron.parameters(), {0.4, -0.6})[0], "Fresh binding after SGD");
}

void validation_checks() {
    using namespace autograd;
    MLP a({2, 4, 1}, 42), same({2, 4, 1}, 42), different({2, 4, 1}, 43);
    require(a.parameters().size() == 17, "Parameter count");
    require(a.parameters() == same.parameters(), "Seed reproducibility");
    require(a.parameters() != different.parameters(), "Distinct seeds");
    rejects([] { MLP invalid({}); });
    rejects([] { MLP invalid({2}); });
    rejects([] { MLP invalid({2, 0, 1}); });
    rejects([] { MLP invalid({std::numeric_limits<std::size_t>::max(), 2}); });
    Tape tape, other;
    auto bound = a.bind(tape);
    rejects([&] { (void)bound({tape.scalar(0)}); });
    rejects([&] { (void)bound({tape.scalar(0), other.scalar(0)}); });
    rejects([&] { BoundMLP invalid(tape, {2, 1}, {1}); });
    rejects([&] { BoundMLP invalid(tape, {0, 1}, {1}); });
    rejects([&] { BoundMLP invalid(tape, {2, 1}, {0, 0, INFINITY}); });
    const auto original = a.parameters();
    rejects([&] { a.set_parameters({0}); });
    auto invalid_params = original;
    invalid_params.back() = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { a.set_parameters(invalid_params); });
    rejects([&] { a.sgd({1}, 0.1); });
    const std::vector<double> zeros(original.size(), 0.0);
    for (double rate : {0.0, -1.0, INFINITY * 1.0, NAN * 1.0})
        rejects([&] { a.sgd(zeros, rate); });
    std::vector<double> gradients(original.size(), 1.0);
    gradients.back() = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { a.sgd(gradients, 0.1); });
    require(a.parameters() == original, "Rejected updates leave all parameters untouched");
    gradients.back() = std::numeric_limits<double>::max();
    rejects<std::overflow_error>([&] { a.sgd(gradients, 2.0); });
    require(a.parameters() == original, "Overflow cannot partially mutate parameters");
    a.sgd(zeros, 0.1);
    require(a.parameters() == original, "Zero-gradient update");
}

void convergence_checks() {
    using namespace autograd;
    constexpr std::array<std::array<double, 3>, 4> rows = {{{-1, -1, -1},
        {-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}};
    for (std::uint32_t seed : {7U, 42U, 2026U}) {
        MLP model({2, 4, 1}, seed);
        for (int step = 0; step < 4000; ++step) {
            Tape tape;
            auto bound = model.bind(tape);
            auto loss = tape.scalar(0);
            for (const auto& row : rows) {
                auto error = bound({tape.scalar(row[0]), tape.scalar(row[1])})[0] - tape.scalar(row[2]);
                loss = loss + error * error;
            }
            loss = loss / tape.scalar(4);
            require(std::isfinite(loss.value()), "Training loss must be finite");
            tape.backward(loss);
            model.sgd(bound.gradients(), 0.1);
        }
        double mse = 0;
        for (const auto& row : rows) {
            const double prediction = reference(model.widths(), model.parameters(), {row[0], row[1]})[0];
            require(std::isfinite(prediction) && (prediction > 0) == (row[2] > 0), "XOR classification");
            mse += (prediction - row[2]) * (prediction - row[2]) / 4;
        }
        require(mse < 0.01, "XOR must reach MSE < 0.01");
        std::cout << "PASS: XOR seed=" << seed << " mse=" << mse << " accuracy=4/4\n";
    }
}
} // namespace

int main() {
    gradient_checks();
    validation_checks();
    convergence_checks();
    std::cout << "PASS: neuron derivatives, all batch parameter gradients, multi-output forward,\n"
                 "deterministic initialization, immutable bindings, SGD, validation, atomic updates\n";
}
