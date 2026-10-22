#include "autograd/linear_classifier_io.hpp"
#include "multiclass_training.hpp"

#include <iostream>

int main() {
    const auto uninterrupted = multiclass_demo::train();
    const auto first_segment = multiclass_demo::train(120);
    const autograd::LinearClassifierModel model{2, 3, first_segment.parameters};
    const std::string checkpoint = autograd::serialize_linear_classifier(model);
    const auto restored = autograd::deserialize_linear_classifier(checkpoint);
    const auto resumed = multiclass_demo::continue_training(restored.parameters, 180);
    const auto holdout = multiclass_demo::evaluate(resumed.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    std::cout << "checkpoint_bytes=" << checkpoint.size()
              << " input_features=" << restored.input_features
              << " classes=" << restored.classes
              << " resume_exact="
              << (resumed.parameters == uninterrupted.parameters ? "yes" : "no")
              << " holdout_accuracy=" << holdout.correct << "/6\n";
    return resumed.parameters == uninterrupted.parameters && holdout.correct == 6 ? 0 : 1;
}
