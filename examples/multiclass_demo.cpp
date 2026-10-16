#include "multiclass_training.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const auto result = multiclass_demo::train();
    const auto training = multiclass_demo::evaluate(result.parameters,
        multiclass_demo::training_points(), multiclass_demo::training_labels());
    const auto holdout = multiclass_demo::evaluate(result.parameters,
        multiclass_demo::holdout_points(), {0, 0, 1, 1, 2, 2});
    std::cout << std::fixed << std::setprecision(9)
              << "Synthetic 2D, three-class linear classifier; 9 parameters\n"
              << "full-batch SGD: steps=300 lr=0.2 zero initialization\n"
              << "initial train cross-entropy=" << result.losses.front() << '\n'
              << "final train cross-entropy=" << training.loss
              << " accuracy=" << training.correct << "/12\n"
              << "disjoint synthetic holdout cross-entropy=" << holdout.loss
              << " accuracy=" << holdout.correct << "/6\n";
    return training.correct == 12 && holdout.correct == 6 && training.loss < 0.02 ? 0 : 1;
}
