#pragma once

#include "autograd/mlp.hpp"
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autograd {
namespace model_io_detail {

constexpr std::size_t kMaxLayers = 64;
constexpr std::size_t kMaxParameters = 10000000;

inline void expect(std::istream& input, const char* expected) {
    std::string token;
    if (!(input >> token) || token != expected)
        throw std::invalid_argument(std::string("Expected '") + expected + "'");
}

inline std::size_t parameter_count(const std::vector<std::size_t>& widths) {
    if (widths.size() < 2 || widths.size() > kMaxLayers)
        throw std::invalid_argument("Model must contain 2 to 64 layer widths");
    std::size_t count = 0;
    for (std::size_t width : widths)
        if (width == 0) throw std::invalid_argument("Layer widths must be positive");
    for (std::size_t layer = 1; layer < widths.size(); ++layer) {
        const std::size_t in = widths[layer - 1], out = widths[layer];
        if (in == std::numeric_limits<std::size_t>::max() ||
            out > (kMaxParameters - count) / (in + 1))
            throw std::invalid_argument("Model exceeds parameter limit");
        count += out * (in + 1);
    }
    return count;
}

} // namespace model_io_detail

// Stable, locale-independent text format. Decimal max_digits10 preserves every
// finite double exactly when parsed by a conforming C++ implementation.
inline void save_model(std::ostream& output, const MLP& model) {
    output.imbue(std::locale::classic());
    output << "AUTOGRAD_MLP 1\nwidths " << model.widths().size();
    for (std::size_t width : model.widths()) output << ' ' << width;
    output << "\nparameters " << model.parameters().size() << "\n";
    output << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double parameter : model.parameters()) output << parameter << '\n';
    output << "end\n";
    if (!output) throw std::runtime_error("Failed to write model");
}

inline MLP load_model(std::istream& input) {
    input.imbue(std::locale::classic());
    model_io_detail::expect(input, "AUTOGRAD_MLP");
    unsigned version = 0;
    if (!(input >> version) || version != 1)
        throw std::invalid_argument("Unsupported model format version");

    model_io_detail::expect(input, "widths");
    std::size_t width_count = 0;
    if (!(input >> width_count) || width_count < 2 ||
        width_count > model_io_detail::kMaxLayers)
        throw std::invalid_argument("Invalid layer count");
    std::vector<std::size_t> widths(width_count);
    for (std::size_t& width : widths)
        if (!(input >> width) || width == 0)
            throw std::invalid_argument("Invalid layer width");
    const std::size_t expected = model_io_detail::parameter_count(widths);

    model_io_detail::expect(input, "parameters");
    std::size_t parameter_count = 0;
    if (!(input >> parameter_count) || parameter_count != expected)
        throw std::invalid_argument("Model parameter count mismatch");
    std::vector<double> parameters(parameter_count);
    for (double& parameter : parameters)
        if (!(input >> parameter) || !std::isfinite(parameter))
            throw std::invalid_argument("Invalid model parameter");
    model_io_detail::expect(input, "end");
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("Trailing model data");
    if (!input.eof()) throw std::invalid_argument("Failed while reading model");

    MLP model(std::move(widths), 0);
    model.set_parameters(parameters);
    return model;
}

inline std::string serialize_model(const MLP& model) {
    std::ostringstream output;
    save_model(output, model);
    return output.str();
}

inline MLP deserialize_model(const std::string& text) {
    std::istringstream input(text);
    return load_model(input);
}

} // namespace autograd
