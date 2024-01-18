#pragma once

#include <nlohmann/json.hpp>

#include "layer.hpp"
#include "matrix.hpp"

namespace tbs {

using json = nlohmann::json;
using Matrices = std::vector<Matrix>;

class NN {
private:
    std::vector<Layer*> l; // layers
    size_t input_size;

public:
    NN(size_t input_layer, Layers& layers);
    NN(json& model);
    ~NN();

    void rand(double min, double max) noexcept;
    void fill(double x) noexcept;
    double cost(Matrices in, Matrices out) noexcept;
    void forward(Matrix& input) noexcept;
    void backprop(NN& g, Matrices in, Matrices out, double rate);
    const Matrix* output() const;

    json save();
    void print() const noexcept;
};

} // namespace tbs
