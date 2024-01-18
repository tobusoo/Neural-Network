#include <layer.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace tbs {

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double relu(double x)
{
    return x ? x > 0 : 0;
}

void Layer::forward(Layer& prev) noexcept
{
    a.mult(prev.a, w);
    a.sum(b);
    a.activation(act_function);
}

Layer::Layer(size_t prev_size, size_t size, ACT_TYPE function_type) noexcept
    : a(1, size), b(1, size), w(prev_size, size), act_type(function_type)
{
    switch (function_type) {
    case RELU:
        act_function = relu;
        break;
    case SIGMOID:
        act_function = sigmoid;
        break;
    default:
        assert(0 && "Unreachable");
        break;
    }
}

Layer::Layer(Layer&& other)
{
    other.a = a;
    other.b = b;
    other.w = w;
    other.act_function = act_function;
    other.act_type = act_type;
}

void Layer::rand(double min, double max) noexcept
{
    b.rand(min, max);
    w.rand(min, max);
}

void Layer::fill(double x) noexcept
{
    b.fill(x);
    w.fill(x);
}

void Layer::forward(Matrix& in) noexcept
{
    a.mult(in, w);
    a.sum(b);
    a.activation(act_function);
}

size_t Layer::size() const noexcept
{
    return a.cols;
}

} // namespace tbs
