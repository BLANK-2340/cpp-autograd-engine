#include "autograd/linear_classifier_io.hpp"
#include "multiclass_training.hpp"

#include <iostream>

int main() {
    const auto trained = multiclass_demo::train();
    const autograd::LinearClassifierModel model{2, 3, trained.parameters};
    const std::string checkpoint = autograd::serialize_linear_classifier(model);
    const auto restored = autograd::deserialize_linear_classifier(checkpoint);
    const auto holdout = multiclass_demo::evaluate(restored.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    std::cout << "checkpoint_bytes=" << checkpoint.size()
              << " input_features=" << restored.input_features
              << " classes=" << restored.classes
              << " holdout_accuracy=" << holdout.correct << "/6\n";
    return holdout.correct == 6 ? 0 : 1;
}
