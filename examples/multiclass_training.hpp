#pragma once

#include "autograd/sgd.hpp"
#include "autograd/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace multiclass_demo {

// Fixed, disjoint synthetic points. Labels follow three clearly separated
// clusters; these are pedagogical data, not an external benchmark.
inline const std::vector<double>& training_points() {
    static const std::vector<double> points{
        -2.0, -1.0, -1.5, -1.5, -2.5, -0.5, -1.5, -0.5,
         2.0, -1.0,  1.5, -1.5,  2.5, -0.5,  1.5, -0.5,
         0.0,  2.0, -0.5,  1.5,  0.5,  2.5,  0.5,  1.5};
    return points;
}

inline const std::vector<std::size_t>& training_labels() {
    static const std::vector<std::size_t> labels{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    return labels;
}

inline const std::vector<double>& holdout_points() {
    static const std::vector<double> points{
        -1.8, -0.8, -2.2, -1.2, 1.8, -0.8, 2.2, -1.2, -0.2, 1.8, 0.2, 2.2};
    return points;
}

struct Evaluation {
    double loss;
    std::size_t correct;
    std::vector<double> probabilities;
};

// Flat layout: six row-major 2x3 weights followed by three shared biases.
inline autograd::Tensor logits(autograd::Tape& tape,
                               const std::vector<double>& points,
                               const std::vector<double>& parameters) {
    if (parameters.size() != 9 || points.size() % 2 != 0)
        throw std::invalid_argument("Classifier requires nine parameters and 2D points");
    for (const double parameter : parameters)
        if (!std::isfinite(parameter))
            throw std::invalid_argument("Classifier parameters must be finite");
    autograd::Tensor input(tape, {points.size() / 2, 2}, points);
    autograd::Tensor weights(tape, {2, 3},
                             std::vector<double>(parameters.begin(), parameters.begin() + 6));
    autograd::Tensor bias(tape, {3},
                          std::vector<double>(parameters.begin() + 6, parameters.end()));
    return autograd::add_rowwise(autograd::matmul(input, weights), bias);
}

inline Evaluation evaluate(const std::vector<double>& parameters,
                           const std::vector<double>& points,
                           const std::vector<std::size_t>& labels) {
    autograd::Tape tape;
    const auto scores = logits(tape, points, parameters);
    const double loss = autograd::cross_entropy(scores, labels).value();
    const auto probabilities = autograd::softmax_rows(scores).values();
    std::size_t correct = 0;
    for (std::size_t row = 0; row < labels.size(); ++row) {
        std::size_t prediction = 0;
        for (std::size_t column = 1; column < 3; ++column)
            if (probabilities[row * 3 + column] > probabilities[row * 3 + prediction])
                prediction = column;
        if (prediction == labels[row]) ++correct;
    }
    return {loss, correct, probabilities};
}

struct TrainingResult {
    std::vector<double> parameters;
    std::vector<double> losses;
};

inline TrainingResult continue_training(const std::vector<double>& initial_parameters,
                                        std::size_t steps,
                                        double learning_rate = 0.2) {
    if (steps == 0 || steps > 10000 || !std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("Training requires 1..10000 steps and a positive finite rate");
    if (initial_parameters.size() != 9)
        throw std::invalid_argument("Classifier training requires nine initial parameters");
    for (const double parameter : initial_parameters)
        if (!std::isfinite(parameter))
            throw std::invalid_argument("Initial classifier parameters must be finite");
    TrainingResult result{initial_parameters, {}};
    result.losses.reserve(steps + 1);
    for (std::size_t step = 0; step < steps; ++step) {
        // The graph is deliberately rebuilt and destroyed once per batch.
        autograd::Tape tape;
        autograd::Tensor input(tape, {12, 2}, training_points());
        autograd::Tensor weights(tape, {2, 3},
            std::vector<double>(result.parameters.begin(), result.parameters.begin() + 6));
        autograd::Tensor bias(tape, {3},
            std::vector<double>(result.parameters.begin() + 6, result.parameters.end()));
        const auto scores = autograd::add_rowwise(autograd::matmul(input, weights), bias);
        const auto loss = autograd::cross_entropy(scores, training_labels());
        result.losses.push_back(loss.value());
        tape.backward(loss);
        auto gradients = weights.gradients();
        const auto bias_gradients = bias.gradients();
        gradients.insert(gradients.end(), bias_gradients.begin(), bias_gradients.end());
        autograd::sgd_step(result.parameters, gradients, learning_rate);
    }
    result.losses.push_back(evaluate(result.parameters, training_points(), training_labels()).loss);
    return result;
}

inline TrainingResult train(std::size_t steps = 300, double learning_rate = 0.2) {
    return continue_training(std::vector<double>(9, 0.0), steps, learning_rate);
}

} // namespace multiclass_demo
