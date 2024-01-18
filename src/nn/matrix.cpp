#include <matrix.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
namespace tbs {

inline static double rand_double(double min, double max) noexcept
{
    return (double)rand() / (double)RAND_MAX * (max - min) + min;
}

template <typename Iter>
[[nodiscard]] Matrix::Matrix(size_t rows, size_t cols, Iter begin, Iter end) noexcept : Matrix(rows, cols)
{
    size_t k = 0;
    for (Iter i = begin; i != end && k < rows * cols; i++, k++) {
        array[k] = *i;
    }
}

[[nodiscard]] Matrix::Matrix(size_t rows, size_t cols) noexcept
    : rows(rows), cols(cols), array(new double[rows * cols])
{
    assert(array && "Could not alloc memory for Matrix");
    memset(array, 0, cols * rows);
}

[[nodiscard]] Matrix::Matrix(size_t rows, size_t cols, std::initializer_list<double> l)
    : Matrix(rows, cols, l.begin(), l.end())
{
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), array(new double[rows * cols])
{
    assert(array && "Could not alloc memory for Matrix");
    std::copy(other.array, other.array + cols * rows, array);
}

Matrix::Matrix(Matrix&& other) : rows(other.rows), cols(other.cols), array(other.array)
{
    other.array = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other)
{
    if (this != &other) {
        delete[] array;
        array = other.array;
        rows = other.rows;
        cols = other.cols;

        other.array = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    return *this;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other) {
        delete[] array;
        rows = other.rows;
        cols = other.cols;
        array = new double[rows * cols];
        assert(array && "Could not alloc memory for Matrix");

        std::copy(other.array, other.array + cols * rows, array);
    }

    return *this;
}

Matrix::~Matrix()
{
    delete array;
}

void Matrix::rand(double min, double max) noexcept
{
    for (size_t i = 0; i < rows * cols; ++i)
        array[i] = rand_double(min, max);
}

void Matrix::fill(double x) noexcept
{
    for (size_t i = 0; i < rows * cols; ++i)
        array[i] = x;
}

void Matrix::sum(Matrix& other) noexcept
{
    assert(other.rows == rows && "The rows of matrices must be equal");
    assert(other.cols == cols && "The cols of matrices must be equal");
    for (size_t i = 0; i < rows * cols; ++i)
        array[i] += other.array[i];
}

void Matrix::mult(Matrix& a, Matrix& b) noexcept
{
    // todo: assert
    size_t i = 0, j = 0, k = 0;
    for (i = 0; i < rows; ++i) {
        double* c = array + i * cols;

        for (j = 0; j < cols; ++j)
            c[j] = 0;

        for (k = 0; k < b.rows; ++k) {
            const double* bb = b.array + k * cols;
            double aa = a.array[i * a.cols + k];

            for (j = 0; j < cols; ++j)
                c[j] += aa * bb[j];
        }
    }
}

void Matrix::activation(double (*act_f)(double)) noexcept
{
    for (size_t i = 0; i < cols * rows; ++i)
        array[i] = act_f(array[i]);
}

void Matrix::print() const noexcept
{
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            printf("%lf ", operator()(i, j));
        printf("\n");
    }
}

[[nodiscard]] double& Matrix::operator()(size_t i, size_t j) const noexcept
{
    return array[i * cols + j];
}

} // namespace tbs
