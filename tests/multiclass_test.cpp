#include "../examples/multiclass_training.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void rejects(Function function) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    require(caught, "Expected validation exception");
}

void update_checks() {
    std::vector<double> parameters{1.0, 2.0, 3.0};
    autograd::sgd_step(parameters, {2.0, -2.0, 0.0}, 0.5);
    require(parameters == std::vector<double>({0.0, 3.0, 3.0}), "Known SGD update mismatch");
    const auto original = parameters;
    rejects<std::invalid_argument>([&] { autograd::sgd_step(parameters, {1.0}, 0.1); });
    rejects<std::invalid_argument>([&] {
        autograd::sgd_step(parameters, {1.0, 1.0, std::numeric_limits<double>::quiet_NaN()}, 0.1);
    });
    rejects<std::invalid_argument>([&] { autograd::sgd_step(parameters, {1, 1, 1}, 0.0); });
    rejects<std::overflow_error>([&] {
        autograd::sgd_step(parameters, {1.0, 1.0, std::numeric_limits<double>::max()}, 2.0);
    });
    require(parameters == original, "Rejected SGD update partially changed parameters");
    autograd::sgd_step(parameters, parameters, 0.5);
    require(parameters == std::vector<double>({0.0, 1.5, 1.5}), "Aliased SGD update mismatch");
}

void convergence_checks() {
    const auto& train_points = multiclass_demo::training_points();
    const auto& test_points = multiclass_demo::holdout_points();
    for (std::size_t test = 0; test < test_points.size(); test += 2) {
        for (std::size_t train = 0; train < train_points.size(); train += 2)
            require(test_points[test] != train_points[train] ||
                        test_points[test + 1] != train_points[train + 1],
                    "Holdout point overlaps a training point");
    }
    const auto first = multiclass_demo::train();
    const auto second = multiclass_demo::train();
    require(first.parameters == second.parameters && first.losses == second.losses,
            "Training is not exactly repeatable in this build");
    require(first.losses.size() == 301, "Training history length mismatch");
    require(std::abs(first.losses.front() - std::log(3.0)) < 1e-12,
            "Zero-initialized loss is not log(3)");
    for (std::size_t index = 0; index < first.losses.size(); ++index) {
        require(std::isfinite(first.losses[index]), "Training loss is not finite");
        if (index > 0)
            require(first.losses[index] <= first.losses[index - 1] + 1e-12,
                    "Default full-batch training loss increased");
    }
    require(first.losses.back() < 0.02, "Training failed to converge");
    const auto training = multiclass_demo::evaluate(first.parameters,
        multiclass_demo::training_points(), multiclass_demo::training_labels());
    const auto holdout = multiclass_demo::evaluate(first.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    require(training.correct == 12 && holdout.correct == 6,
            "Synthetic training or holdout classification failed");
    for (const double probability : holdout.probabilities)
        require(std::isfinite(probability) && probability >= 0.0 && probability <= 1.0,
                "Holdout probability is invalid");
    rejects<std::invalid_argument>([] { (void)multiclass_demo::train(0); });
    rejects<std::invalid_argument>([] { (void)multiclass_demo::train(10001); });
    rejects<std::invalid_argument>([] { (void)multiclass_demo::train(1, -1.0); });
}
} // namespace

int main() {
    update_checks();
    convergence_checks();
    std::cout << "PASS: synthetic multiclass convergence, repeatability, holdout, atomic SGD\n";
}
