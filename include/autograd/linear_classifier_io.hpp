#pragma once

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autograd {

struct LinearClassifierModel {
    std::size_t input_features;
    std::size_t classes;
    std::vector<double> parameters;
};

namespace linear_classifier_io_detail {

constexpr std::size_t kMaxParameters = 10000000;

inline void expect(std::istream& input, const char* expected) {
    std::string token;
    if (!(input >> token) || token != expected)
        throw std::invalid_argument(std::string("Expected '") + expected + "'");
}

inline std::size_t parameter_count(std::size_t input_features, std::size_t classes) {
    if (input_features == 0 || classes < 2)
        throw std::invalid_argument("Classifier dimensions must be positive with at least two classes");
    if (input_features == std::numeric_limits<std::size_t>::max() ||
        classes > kMaxParameters / (input_features + 1))
        throw std::invalid_argument("Classifier exceeds parameter limit");
    return (input_features + 1) * classes;
}

inline void validate(const LinearClassifierModel& model) {
    const std::size_t expected = parameter_count(model.input_features, model.classes);
    if (model.parameters.size() != expected)
        throw std::invalid_argument("Classifier parameter count mismatch");
    for (const double parameter : model.parameters)
        if (!std::isfinite(parameter))
            throw std::invalid_argument("Classifier parameters must be finite");
}

} // namespace linear_classifier_io_detail

// Stable, locale-independent text checkpoint. Parameters are row-major weights
// followed by one bias per class.
inline void save_linear_classifier(std::ostream& output,
                                   const LinearClassifierModel& model) {
    linear_classifier_io_detail::validate(model);
    output.imbue(std::locale::classic());
    output << "AUTOGRAD_LINEAR_CLASSIFIER 1\n"
           << "input_features " << model.input_features << '\n'
           << "classes " << model.classes << '\n'
           << "parameters " << model.parameters.size() << '\n'
           << std::scientific
           << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const double parameter : model.parameters) output << parameter << '\n';
    output << "end\n";
    if (!output) throw std::runtime_error("Failed to write linear classifier");
}

inline LinearClassifierModel load_linear_classifier(std::istream& input) {
    input.imbue(std::locale::classic());
    linear_classifier_io_detail::expect(input, "AUTOGRAD_LINEAR_CLASSIFIER");
    unsigned version = 0;
    if (!(input >> version) || version != 1)
        throw std::invalid_argument("Unsupported linear-classifier format version");

    linear_classifier_io_detail::expect(input, "input_features");
    std::size_t input_features = 0;
    if (!(input >> input_features))
        throw std::invalid_argument("Invalid input-feature count");
    linear_classifier_io_detail::expect(input, "classes");
    std::size_t classes = 0;
    if (!(input >> classes)) throw std::invalid_argument("Invalid class count");
    const std::size_t expected =
        linear_classifier_io_detail::parameter_count(input_features, classes);

    linear_classifier_io_detail::expect(input, "parameters");
    std::size_t parameter_count = 0;
    if (!(input >> parameter_count) || parameter_count != expected)
        throw std::invalid_argument("Classifier parameter count mismatch");
    std::vector<double> parameters(parameter_count);
    for (double& parameter : parameters)
        if (!(input >> parameter) || !std::isfinite(parameter))
            throw std::invalid_argument("Invalid classifier parameter");
    linear_classifier_io_detail::expect(input, "end");
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("Trailing classifier data");
    if (!input.eof()) throw std::invalid_argument("Failed while reading linear classifier");

    return {input_features, classes, std::move(parameters)};
}

inline std::string serialize_linear_classifier(const LinearClassifierModel& model) {
    std::ostringstream output;
    save_linear_classifier(output, model);
    return output.str();
}

inline LinearClassifierModel deserialize_linear_classifier(const std::string& text) {
    std::istringstream input(text);
    return load_linear_classifier(input);
}

} // namespace autograd
