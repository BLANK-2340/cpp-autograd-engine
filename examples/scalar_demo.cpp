#include "autograd/scalar.hpp"
#include <iostream>

int main() {
    autograd::Tape tape;
    auto x = tape.scalar(3.0);
    auto y = tape.scalar(2.0);
    auto output = x * x + x * y;
    tape.backward(output);
    std::cout << "f = x*x + x*y\nf = " << output.value()
              << "\ndf/dx = " << x.grad() << "\ndf/dy = " << y.grad() << '\n';
}
