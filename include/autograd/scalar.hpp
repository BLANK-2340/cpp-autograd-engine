#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace autograd {
class Tape;

// Scalars are handles; their Tape must outlive them.
class Scalar {
    friend class Tape;
    Tape* tape_;
    std::size_t id_;
    Scalar(Tape* tape, std::size_t id) : tape_(tape), id_(id) {}
public:
    double value() const;
    double grad() const;
    Tape& tape() const { return *tape_; }
    std::size_t id() const { return id_; }
};

// Append-only graph: each edge stores its forward-time local derivative.
// Rebuild a fresh tape for each training step after updating parameters.
class Tape {
    struct Edge { std::size_t parent; double derivative; };
    struct Node { double value; double grad; std::vector<Edge> edges; };
    std::vector<Node> nodes_;
public:
    Tape() = default;
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;
    Tape(Tape&&) = delete;
    Tape& operator=(Tape&&) = delete;

    Scalar scalar(double value) {
        nodes_.push_back({value, 0.0, {}});
        return Scalar(this, nodes_.size() - 1);
    }
    Scalar unary(Scalar a, double value, double derivative) {
        require(a);
        nodes_.push_back({value, 0.0, {{a.id_, derivative}}});
        return Scalar(this, nodes_.size() - 1);
    }
    Scalar binary(Scalar a, Scalar b, double value, double da, double db) {
        require(a); require(b);
        nodes_.push_back({value, 0.0, {{a.id_, da}, {b.id_, db}}});
        return Scalar(this, nodes_.size() - 1);
    }
    void require(Scalar x) const {
        if (x.tape_ != this) throw std::invalid_argument("Scalars belong to different tapes");
    }
    double value(Scalar x) const { require(x); return nodes_.at(x.id_).value; }
    double grad(Scalar x) const { require(x); return nodes_.at(x.id_).grad; }
    void backward(Scalar output, double seed = 1.0) {
        require(output);
        for (auto& node : nodes_) node.grad = 0.0;
        nodes_[output.id_].grad = seed;
        // All parents precede their children, avoiding recursion even on deep graphs.
        for (std::size_t i = output.id_ + 1; i-- > 0;) {
            const double g = nodes_[i].grad;
            if (g == 0.0) continue;
            for (const auto& edge : nodes_[i].edges)
                nodes_[edge.parent].grad += g * edge.derivative;
        }
    }
};

inline double Scalar::value() const { return tape_->value(*this); }
inline double Scalar::grad() const { return tape_->grad(*this); }
inline Scalar operator+(Scalar a, Scalar b) {
    return a.tape().binary(a, b, a.value() + b.value(), 1.0, 1.0);
}
inline Scalar operator-(Scalar a, Scalar b) {
    return a.tape().binary(a, b, a.value() - b.value(), 1.0, -1.0);
}
inline Scalar operator*(Scalar a, Scalar b) {
    return a.tape().binary(a, b, a.value() * b.value(), b.value(), a.value());
}
inline Scalar operator/(Scalar a, Scalar b) {
    a.tape().require(b);
    if (b.value() == 0.0) throw std::domain_error("Division by zero");
    return a.tape().binary(a, b, a.value() / b.value(),
                           1.0 / b.value(), -a.value() / (b.value() * b.value()));
}
inline Scalar operator-(Scalar a) { return a.tape().unary(a, -a.value(), -1.0); }
inline Scalar tanh(Scalar a) {
    const double t = std::tanh(a.value());
    return a.tape().unary(a, t, 1.0 - t * t);
}
inline Scalar exp(Scalar a) {
    const double e = std::exp(a.value());
    return a.tape().unary(a, e, e);
}
inline Scalar log(Scalar a) {
    if (a.value() <= 0.0) throw std::domain_error("Log requires a positive value");
    return a.tape().unary(a, std::log(a.value()), 1.0 / a.value());
}
inline Scalar relu(Scalar a) {
    return a.tape().unary(a, a.value() > 0.0 ? a.value() : 0.0,
                         a.value() > 0.0 ? 1.0 : 0.0);
}
} // namespace autograd
