#include "autograd/model_io.hpp"
#include <iostream>
#include <sstream>

int main() {
    autograd::MLP model({2, 4, 1}, 42);
    std::stringstream checkpoint;
    autograd::save_model(checkpoint, model);
    const auto restored = autograd::load_model(checkpoint);
    std::cout << "format=AUTOGRAD_MLP version=1 widths=";
    for (std::size_t i = 0; i < restored.widths().size(); ++i)
        std::cout << (i ? "x" : "") << restored.widths()[i];
    std::cout << " parameters=" << restored.parameters().size()
              << " exact=" << (restored.parameters() == model.parameters() ? "yes" : "no") << '\n';
    if (restored.parameters() != model.parameters())
        throw std::runtime_error("Model round trip failed");
}
