#include "autograd/scalar.hpp"
#include <iostream>
#include <string>

void near(double got, double expected, const std::string& name) {
    if (!std::isfinite(got) || std::abs(got - expected) > 1e-6 * (1 + std::abs(expected)))
        throw std::runtime_error(name + " gradient mismatch");
}

int main() {
    using namespace autograd;
    Tape t;
    auto x = t.scalar(0.7), y = t.scalar(1.3);
    auto f = log(exp(tanh(x * y + x / y - y)));
    t.backward(f);
    auto reference = [](double a, double b) { return std::tanh(a*b + a/b - b); };
    const double h = 1e-5;
    near(x.grad(), (reference(0.7+h, 1.3)-reference(0.7-h, 1.3))/(2*h), "composite x");
    near(y.grad(), (reference(0.7, 1.3+h)-reference(0.7, 1.3-h))/(2*h), "composite y");
    auto square = x * x;
    auto shared = square + square + x;
    t.backward(shared);
    near(x.grad(), 4*0.7 + 1, "shared graph");
    t.backward(shared);
    near(x.grad(), 4*0.7 + 1, "repeated backward resets");
    t.backward(square, 2.0);
    near(x.grad(), 4*0.7, "seed");
    near(y.grad(), 0, "disconnected node");
    auto negative = -x;
    t.backward(negative);
    near(x.grad(), -1, "negation");
    auto zero = t.scalar(0), positive = t.scalar(2), below = t.scalar(-2);
    auto activation = relu(zero) + relu(positive) + relu(below);
    t.backward(activation);
    near(zero.grad(), 0, "relu zero convention");
    near(positive.grad(), 1, "relu positive");
    near(below.grad(), 0, "relu negative");
    int caught = 0;
    try { (void)(x / zero); } catch (const std::domain_error&) { ++caught; }
    try { (void)log(below); } catch (const std::domain_error&) { ++caught; }
    Tape other;
    try { (void)(x + other.scalar(1)); } catch (const std::invalid_argument&) { ++caught; }
    if (caught != 3) throw std::runtime_error("Domain/tape guards failed");
    auto chain = x;
    for (int i = 0; i < 100000; ++i) chain = chain + zero;
    t.backward(chain);
    near(x.grad(), 1, "deep graph");
    std::cout << "PASS: finite differences, shared paths, reset, seed, disconnected nodes,\n"
                 "negation, ReLU boundaries, domain guards, tape isolation, 100000-node chain\n";
}
