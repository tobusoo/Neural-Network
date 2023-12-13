#pragma once
#include <cassert>
#include <nlohmann/json.hpp>
#include <vector>

#include <nn/matrix.h>

using json = nlohmann::json;

double sigmoid(double x);

#define ARRAY_LEN(ar) (sizeof((ar)) / sizeof((ar)[0]))

class NN {
private:
    using MatrixArray = std::vector<Matrix>;

    size_t* m_layers;
    size_t m_layers_count;

    MatrixArray w; // layers_count - 1
    MatrixArray b; // layers_count - 1
    MatrixArray a; // layers_count
public:
    NN(const size_t* layers, size_t layers_count);
    NN(json& model);
    ~NN();

    const size_t* get_layers() const noexcept;
    size_t get_layers_count() const noexcept;
    const Matrix& get_out() const noexcept;

    void rand(double min, double max) noexcept;
    void fill(double x) noexcept;
    void forward(Matrix& input) noexcept;
    double cost(MatrixArray in, MatrixArray out) noexcept;
    void backprop(NN& gradient, MatrixArray in, MatrixArray out, double rate);

    // this function is intended only for the classification problem
    // when the expected output vector looks like [0, 1, 0, 0]
    double cost(MatrixArray in, MatrixArray out, double& correct) noexcept;

    json save();
};