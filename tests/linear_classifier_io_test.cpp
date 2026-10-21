#include "autograd/linear_classifier_io.hpp"
#include "../examples/multiclass_training.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function) {
    bool caught = false;
    try { function(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Malformed classifier checkpoint was accepted");
}

void exact_round_trip() {
    autograd::LinearClassifierModel model{2, 3, {
        0.0, -0.0, std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), -1.25,
        2.5, -3.75, 4.125}};
    const std::string encoded = autograd::serialize_linear_classifier(model);
    const auto restored = autograd::deserialize_linear_classifier(encoded);
    require(restored.input_features == model.input_features && restored.classes == model.classes,
            "Classifier dimensions changed during round trip");
    require(restored.parameters == model.parameters,
            "Classifier parameters changed during round trip");
    require(std::signbit(restored.parameters[1]), "Negative zero was not preserved");
    require(autograd::serialize_linear_classifier(restored) == encoded,
            "Classifier serialization is not canonical");

    std::stringstream stream;
    autograd::save_linear_classifier(stream, model);
    const auto streamed = autograd::load_linear_classifier(stream);
    require(streamed.parameters == model.parameters, "Stream round trip changed parameters");
}

void trained_prediction_round_trip() {
    const auto trained = multiclass_demo::train();
    autograd::LinearClassifierModel model{2, 3, trained.parameters};
    const auto before = multiclass_demo::evaluate(model.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    const auto restored = autograd::deserialize_linear_classifier(
        autograd::serialize_linear_classifier(model));
    const auto after = multiclass_demo::evaluate(restored.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    require(after.loss == before.loss && after.correct == before.correct &&
                after.probabilities == before.probabilities,
            "Checkpoint round trip changed classifier predictions");
}

void malformed_inputs() {
    const std::vector<std::string> invalid{
        "", "OTHER 1\n", "AUTOGRAD_LINEAR_CLASSIFIER 2\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 0\nclasses 3\nparameters 3\n0\n0\n0\nend\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 2\nclasses 1\nparameters 3\n0\n0\n0\nend\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 2\nclasses 3\nparameters 8\nend\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 2\nclasses 3\nparameters 9\n0\n0\n0\n0\n0\n0\n0\n0\nend\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 2\nclasses 3\nparameters 9\n0\n0\n0\n0\n0\n0\n0\n0\nnan\nend\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 2\nclasses 3\nparameters 9\n0\n0\n0\n0\n0\n0\n0\n0\n0\nend\njunk\n",
        "AUTOGRAD_LINEAR_CLASSIFIER 1\ninput_features 10000000\nclasses 2\nparameters 0\nend\n"
    };
    for (const std::string& text : invalid)
        rejects([&] { (void)autograd::deserialize_linear_classifier(text); });

    rejects([] {
        autograd::LinearClassifierModel bad{2, 3, std::vector<double>(8, 0.0)};
        (void)autograd::serialize_linear_classifier(bad);
    });
    rejects([] {
        autograd::LinearClassifierModel bad{2, 3, std::vector<double>(9, 0.0)};
        bad.parameters[4] = std::numeric_limits<double>::infinity();
        (void)autograd::serialize_linear_classifier(bad);
    });
}

} // namespace

int main() {
    exact_round_trip();
    trained_prediction_round_trip();
    malformed_inputs();
    std::cout << "PASS: exact classifier checkpoint round trips, prediction preservation, "
                 "and malformed input rejection\n";
}
