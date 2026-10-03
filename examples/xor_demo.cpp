#include "autograd/mlp.hpp"
#include <array>
#include <iomanip>
#include <iostream>

int main() {
    using namespace autograd;
    constexpr std::array<std::array<double, 3>, 4> data = {{{-1, -1, -1},
        {-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}};
    MLP model({2, 4, 1}, 42);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "XOR: 2-4-1 tanh MLP, seed=42, full-batch SGD, lr=0.1\n";
    for (int step = 0; step <= 4000; ++step) {
        Tape tape;
        auto network = model.bind(tape);
        auto loss = tape.scalar(0.0);
        for (const auto& row : data) {
            const auto prediction = network({tape.scalar(row[0]), tape.scalar(row[1])})[0];
            const auto error = prediction - tape.scalar(row[2]);
            loss = loss + error * error;
        }
        loss = loss / tape.scalar(static_cast<double>(data.size()));
        if (step % 1000 == 0) std::cout << "step=" << step << " mse=" << loss.value() << '\n';
        if (step < 4000) {
            tape.backward(loss);
            model.sgd(network.gradients(), 0.1);
        } else if (!std::isfinite(loss.value()) || loss.value() >= 0.01) {
            throw std::runtime_error("XOR failed to converge");
        }
    }
    Tape tape;
    auto network = model.bind(tape);
    int correct = 0;
    for (const auto& row : data) {
        const auto prediction = network({tape.scalar(row[0]), tape.scalar(row[1])})[0].value();
        if (std::isfinite(prediction) && (prediction > 0) == (row[2] > 0)) ++correct;
        std::cout << '(' << row[0] << ", " << row[1] << ") target=" << row[2]
                  << " prediction=" << prediction << '\n';
    }
    std::cout << "accuracy=" << correct << "/4\n";
    if (correct != 4) throw std::runtime_error("XOR predictions failed");
}
