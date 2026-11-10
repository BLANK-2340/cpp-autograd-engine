#include "autograd/linear_classifier_io.hpp"
#include "../examples/multiclass_training.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
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

void exact_resume_training() {
    const auto uninterrupted = multiclass_demo::train(300, 0.2);
    const auto first_segment = multiclass_demo::train(120, 0.2);
    const autograd::LinearClassifierModel checkpoint{2, 3, first_segment.parameters};
    const auto restored = autograd::deserialize_linear_classifier(
        autograd::serialize_linear_classifier(checkpoint));
    const auto second_segment = multiclass_demo::continue_training(
        restored.parameters, 180, 0.2);

    std::vector<double> resumed_losses = first_segment.losses;
    resumed_losses.insert(resumed_losses.end(), second_segment.losses.begin() + 1,
                          second_segment.losses.end());
    require(second_segment.parameters == uninterrupted.parameters,
            "Resumed parameters differ from uninterrupted training");
    require(resumed_losses == uninterrupted.losses,
            "Resumed loss trajectory differs from uninterrupted training");

    rejects([] { (void)multiclass_demo::continue_training({}, 1); });
    rejects([] {
        auto parameters = std::vector<double>(9, 0.0);
        parameters[3] = std::numeric_limits<double>::quiet_NaN();
        (void)multiclass_demo::continue_training(parameters, 1);
    });
    rejects([] { (void)multiclass_demo::continue_training(std::vector<double>(9), 0); });
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

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to read classifier test checkpoint");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void atomic_path_save_and_resume() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "autograd-classifier-io-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path checkpoint = directory / "classifier.aglc";
    const std::filesystem::path temporary = checkpoint.string() + ".tmp";

    const auto uninterrupted = multiclass_demo::train(300, 0.2);
    const auto first_segment = multiclass_demo::train(120, 0.2);
    const autograd::LinearClassifierModel first{2, 3, first_segment.parameters};
    autograd::save_linear_classifier_atomic(checkpoint, first);
    require(read_file(checkpoint) == autograd::serialize_linear_classifier(first),
            "Atomic classifier save changed canonical bytes");
    require(!std::filesystem::exists(temporary),
            "Successful classifier save left a temporary file");

    std::ifstream input(checkpoint, std::ios::binary);
    const auto restored = autograd::load_linear_classifier(input);
    const auto second_segment = multiclass_demo::continue_training(
        restored.parameters, 180, 0.2);
    std::vector<double> resumed_losses = first_segment.losses;
    resumed_losses.insert(resumed_losses.end(), second_segment.losses.begin() + 1,
                          second_segment.losses.end());
    require(second_segment.parameters == uninterrupted.parameters &&
                resumed_losses == uninterrupted.losses,
            "Atomic classifier checkpoint changed exact resume behavior");

    const autograd::LinearClassifierModel replacement{2, 3,
                                                       uninterrupted.parameters};
    autograd::save_linear_classifier_atomic(checkpoint, replacement);
    const std::string published = autograd::serialize_linear_classifier(replacement);
    require(read_file(checkpoint) == published,
            "Atomic classifier replacement published incorrect bytes");

    autograd::LinearClassifierModel invalid = replacement;
    invalid.parameters.pop_back();
    bool write_failed = false;
    try {
        autograd::save_linear_classifier_atomic(checkpoint, invalid);
    } catch (const std::invalid_argument&) {
        write_failed = true;
    }
    require(write_failed, "Invalid classifier path save was accepted");
    require(read_file(checkpoint) == published &&
                !std::filesystem::exists(temporary),
            "Failed classifier write changed the destination or left a temporary");

    bool replacement_failed = false;
    try {
        autograd::atomic_file_detail::write_file(
            checkpoint,
            [&](std::ostream& output) {
                autograd::save_linear_classifier(output, first);
            },
            [](const std::filesystem::path&, const std::filesystem::path&) {
                throw std::runtime_error("injected classifier replacement failure");
            });
    } catch (const std::runtime_error&) {
        replacement_failed = true;
    }
    require(replacement_failed, "Injected classifier replacement failure was not reported");
    require(read_file(checkpoint) == published &&
                !std::filesystem::exists(temporary),
            "Failed classifier replacement changed the destination or left a temporary");

    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    exact_round_trip();
    trained_prediction_round_trip();
    exact_resume_training();
    malformed_inputs();
    atomic_path_save_and_resume();
    std::cout << "PASS: exact classifier checkpoint round trips, prediction preservation, "
                 "resume equivalence, atomic path failure safety, and malformed input "
                 "rejection\n";
}
