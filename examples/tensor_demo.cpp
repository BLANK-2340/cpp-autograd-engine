#include "autograd/tensor.hpp"

#include <iostream>

int main() {
    using namespace autograd;
    Tape tape;
    Tensor input(tape, {1, 2}, {2.0, -1.0});
    Tensor weights(tape, {2, 2}, {0.5, 1.0, -0.25, 0.75});
    const Tensor output = tanh(matmul(input, weights));
    const Scalar loss = sum(output);
    tape.backward(loss);

    std::cout << "output:";
    for (const double value : output.values()) std::cout << ' ' << value;
    std::cout << "\ninput gradients:";
    for (const double gradient : input.gradients()) std::cout << ' ' << gradient;
    std::cout << '\n';
}
