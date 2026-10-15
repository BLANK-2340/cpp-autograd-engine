#include "autograd/tensor.hpp"

#include <iomanip>
#include <iostream>

int main() {
    using namespace autograd;
    Tape tape;
    Tensor logits(tape, {2, 3}, {2.0, 0.5, -1.0, -0.25, 0.75, 1.5});
    const Tensor probabilities = softmax_rows(logits);
    const Scalar loss = cross_entropy(logits, {0, 2});
    tape.backward(loss);

    std::cout << std::fixed << std::setprecision(6) << "probabilities:\n";
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 3; ++column)
            std::cout << (column == 0 ? "" : " ")
                      << probabilities.at(row, column).value();
        std::cout << '\n';
    }
    std::cout << "mean cross-entropy: " << loss.value() << "\nlogit gradients:\n";
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 3; ++column)
            std::cout << (column == 0 ? "" : " ")
                      << logits.at(row, column).grad();
        std::cout << '\n';
    }
}
