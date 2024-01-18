#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "matrix.hpp"
namespace tbs {

double sigmoid(double x);
double relu(double x);

enum ACT_TYPE { RELU, SIGMOID }; // type of activation func

struct LayerWrapper {
    size_t size;
    ACT_TYPE act_type;
};

using Layers = std::vector<LayerWrapper>;

class Layer {
    friend class NN;
    using ACT_F = double(double); // activation func

private:
    Matrix a; // matrix of activation neurons
    Matrix b; // matrix of biases
    Matrix w; // matrix of weights

    ACT_F* act_function; // pointer to activation function
    ACT_TYPE act_type;   // type of activation function

    void forward(Layer& prev) noexcept;

public:
    Layer(size_t prev_size, size_t size, ACT_TYPE function_type) noexcept;
    Layer(Layer&& other);
    ~Layer() = default;

    void rand(double min, double max) noexcept;
    void fill(double x) noexcept;
    void forward(Matrix& in) noexcept;

    size_t size() const noexcept;
};
} // namespace tbs
