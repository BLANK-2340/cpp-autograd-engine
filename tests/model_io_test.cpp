#include "autograd/model_io.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& fn) {
    bool caught = false;
    try { fn(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Malformed model was accepted");
}

std::vector<double> predict(const autograd::MLP& model, const std::vector<double>& values) {
    autograd::Tape tape;
    auto bound = model.bind(tape);
    std::vector<autograd::Scalar> input;
    for (double value : values) input.push_back(tape.scalar(value));
    std::vector<double> result;
    for (auto output : bound(input)) result.push_back(output.value());
    return result;
}

void exact_round_trip() {
    autograd::MLP model({2, 3, 2}, 2026);
    auto parameters = model.parameters();
    parameters[0] = 0.0;
    parameters[1] = -0.0;
    parameters[2] = std::numeric_limits<double>::denorm_min();
    parameters[3] = std::numeric_limits<double>::min();
    parameters[4] = std::numeric_limits<double>::max();
    model.set_parameters(parameters);

    const std::string encoded = autograd::serialize_model(model);
    const auto restored = autograd::deserialize_model(encoded);
    require(restored.widths() == model.widths(), "Architecture round trip");
    require(restored.parameters() == model.parameters(), "Parameter round trip");
    require(std::signbit(restored.parameters()[1]), "Negative zero round trip");
    require(autograd::serialize_model(restored) == encoded, "Canonical serialization");

    autograd::MLP ordinary({2, 4, 1}, 42);
    const auto before = predict(ordinary, {0.25, -0.75});
    const auto loaded = autograd::deserialize_model(autograd::serialize_model(ordinary));
    require(predict(loaded, {0.25, -0.75}) == before, "Prediction round trip");

    std::stringstream stream;
    autograd::save_model(stream, ordinary);
    const auto streamed = autograd::load_model(stream);
    require(streamed.parameters() == ordinary.parameters(), "Stream API round trip");
}

void malformed_inputs() {
    const std::vector<std::string> invalid = {
        "", "OTHER 1\n", "AUTOGRAD_MLP 2\n",
        "AUTOGRAD_MLP 1\nwidths 1 2\nparameters 0\nend\n",
        "AUTOGRAD_MLP 1\nwidths 2 0 1\nparameters 1\n0\nend\n",
        "AUTOGRAD_MLP 1\nwidths 2 2 1\nparameters 2\n0\n0\nend\n",
        "AUTOGRAD_MLP 1\nwidths 2 2 1\nparameters 3\n0\n0\n",
        "AUTOGRAD_MLP 1\nwidths 2 2 1\nparameters 3\n0\n0\nnan\nend\n",
        "AUTOGRAD_MLP 1\nwidths 2 2 1\nparameters 3\n0\n0\n0\nend\njunk\n",
        "AUTOGRAD_MLP 1\nwidths 2 10000001 10000001\nparameters 0\nend\n"
    };
    for (const auto& text : invalid)
        rejects([&] { (void)autograd::deserialize_model(text); });

    std::string too_many = "AUTOGRAD_MLP 1\nwidths 65";
    for (int i = 0; i < 65; ++i) too_many += " 1";
    too_many += "\nparameters 0\nend\n";
    rejects([&] { (void)autograd::deserialize_model(too_many); });
}
} // namespace

int main() {
    exact_round_trip();
    malformed_inputs();
    std::cout << "PASS: exact model round trips, canonical output, prediction preservation, "
                 "stream API, malformed and oversized input rejection\n";
}
