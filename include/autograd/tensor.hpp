#pragma once

#include "autograd/scalar.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autograd {

// A small dense row-major tensor whose elements are Scalar handles. The tape
// owns the graph; a Tensor is only a shaped view over handles and must not
// outlive its tape.
class Tensor {
    static constexpr std::size_t kMaxRank = 8;
    static constexpr std::size_t kMaxElements = 1000000;

    Tape* tape_;
    std::vector<std::size_t> shape_;
    std::vector<Scalar> elements_;

    struct ScalarElementsTag {};

    Tensor(Tape& tape, std::vector<std::size_t> shape,
           std::vector<Scalar> elements, ScalarElementsTag)
        : tape_(&tape), shape_(std::move(shape)), elements_(std::move(elements)) {
        if (elements_.size() != checked_numel(shape_))
            throw std::invalid_argument("Tensor element count does not match shape");
        for (const Scalar element : elements_) tape.require(element);
    }

    static std::size_t checked_numel(const std::vector<std::size_t>& shape) {
        if (shape.empty() || shape.size() > kMaxRank)
            throw std::invalid_argument("Tensor rank must be between 1 and 8");
        std::size_t count = 1;
        bool zero = false;
        for (const std::size_t dimension : shape) {
            if (dimension > kMaxElements)
                throw std::invalid_argument("Tensor dimension exceeds resource limit");
            if (dimension == 0) {
                zero = true;
            } else if (!zero) {
                if (count > kMaxElements / dimension)
                    throw std::invalid_argument("Tensor exceeds element limit");
                count *= dimension;
            }
        }
        return zero ? 0 : count;
    }

public:
    Tensor(Tape& tape, std::vector<std::size_t> shape,
           const std::vector<double>& values)
        : tape_(&tape), shape_(std::move(shape)) {
        const std::size_t count = checked_numel(shape_);
        if (values.size() != count)
            throw std::invalid_argument("Tensor value count does not match shape");
        elements_.reserve(count);
        for (const double value : values) elements_.push_back(tape.scalar(value));
    }

    static Tensor from_scalars(Tape& tape, std::vector<std::size_t> shape,
                               std::vector<Scalar> elements) {
        return Tensor(tape, std::move(shape), std::move(elements), ScalarElementsTag{});
    }

    Tape& tape() const { return *tape_; }
    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return elements_.size(); }
    bool empty() const { return elements_.empty(); }

    Scalar operator[](std::size_t index) const { return elements_.at(index); }

    Scalar at(std::size_t row, std::size_t column) const {
        if (rank() != 2) throw std::invalid_argument("Two indices require a matrix");
        if (row >= shape_[0] || column >= shape_[1])
            throw std::out_of_range("Tensor index out of range");
        return elements_[row * shape_[1] + column];
    }

    std::vector<double> values() const {
        std::vector<double> result;
        result.reserve(size());
        for (const Scalar element : elements_) result.push_back(element.value());
        return result;
    }

    std::vector<double> gradients() const {
        std::vector<double> result;
        result.reserve(size());
        for (const Scalar element : elements_) result.push_back(element.grad());
        return result;
    }
};

inline void require_compatible(const Tensor& lhs, const Tensor& rhs) {
    if (&lhs.tape() != &rhs.tape())
        throw std::invalid_argument("Tensors belong to different tapes");
    if (lhs.shape() != rhs.shape())
        throw std::invalid_argument("Tensor shapes differ");
}

inline Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    require_compatible(lhs, rhs);
    std::vector<Scalar> result;
    result.reserve(lhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index)
        result.push_back(lhs[index] + rhs[index]);
    return Tensor::from_scalars(lhs.tape(), lhs.shape(), std::move(result));
}

inline Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    require_compatible(lhs, rhs);
    std::vector<Scalar> result;
    result.reserve(lhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index)
        result.push_back(lhs[index] * rhs[index]);
    return Tensor::from_scalars(lhs.tape(), lhs.shape(), std::move(result));
}

inline Tensor tanh(const Tensor& input) {
    std::vector<Scalar> result;
    result.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
        result.push_back(autograd::tanh(input[index]));
    return Tensor::from_scalars(input.tape(), input.shape(), std::move(result));
}

inline Tensor relu(const Tensor& input) {
    std::vector<Scalar> result;
    result.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
        result.push_back(autograd::relu(input[index]));
    return Tensor::from_scalars(input.tape(), input.shape(), std::move(result));
}

inline Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (&lhs.tape() != &rhs.tape())
        throw std::invalid_argument("Tensors belong to different tapes");
    if (lhs.rank() != 2 || rhs.rank() != 2)
        throw std::invalid_argument("Matrix multiplication requires rank-2 tensors");
    const std::size_t rows = lhs.shape()[0];
    const std::size_t inner = lhs.shape()[1];
    const std::size_t columns = rhs.shape()[1];
    if (inner != rhs.shape()[0])
        throw std::invalid_argument("Matrix inner dimensions differ");

    std::vector<Scalar> result;
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::invalid_argument("Matrix output size overflows");
    result.reserve(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            Scalar total = lhs.tape().scalar(0.0);
            for (std::size_t index = 0; index < inner; ++index)
                total = total + lhs.at(row, index) * rhs.at(index, column);
            result.push_back(total);
        }
    }
    return Tensor::from_scalars(lhs.tape(), {rows, columns}, std::move(result));
}

inline Scalar sum(const Tensor& input) {
    Scalar total = input.tape().scalar(0.0);
    for (std::size_t index = 0; index < input.size(); ++index)
        total = total + input[index];
    return total;
}

inline Scalar mean(const Tensor& input) {
    if (input.empty()) throw std::domain_error("Mean requires a non-empty tensor");
    return sum(input) / input.tape().scalar(static_cast<double>(input.size()));
}

} // namespace autograd
