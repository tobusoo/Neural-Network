#pragma once

#include <cstddef>
#include <initializer_list>

namespace tbs {

struct Matrix {
    size_t rows = 0;
    size_t cols = 0;
    double* array = nullptr;

    Matrix() = default;
    template <typename Iter>
    [[nodiscard]] Matrix(size_t rows, size_t cols, Iter begin, Iter end) noexcept;
    [[nodiscard]] Matrix(size_t rows, size_t cols) noexcept;
    [[nodiscard]] Matrix(size_t rows, size_t cols, std::initializer_list<double> l);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    void rand(double min, double max) noexcept;
    void fill(double x) noexcept;
    void sum(Matrix& other) noexcept;
    void mult(Matrix& a, Matrix& b) noexcept;
    void activation(double (*act_f)(double)) noexcept;
    void print() const noexcept;

    [[nodiscard]] double& operator()(size_t i, size_t j) const noexcept;
};
} // namespace tbs
