#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <nn/matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double rand_double(double low, double high)
{
    return (double)rand() / (double)RAND_MAX * (high - low) + low;
}

Matrix matrix_alloc(size_t rows, size_t cols)
{
    Matrix matrix = {.m = NULL, .rows = rows, .cols = cols};

    matrix.m = (double*)malloc(sizeof(*matrix.m) * rows * cols);
    assert(matrix.m != NULL
           && "matrix_alloc ERROR: not enough memory. Buy more!");

    return matrix;
}

void matrix_rand(Matrix m, double low, double high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MATRIX_AT(m, i, j) = rand_double(low, high);
        }
    }
}

void matrix_fill(Matrix m, double x)
{
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.m[i] = x;
}

void matrix_sum_dst(Matrix dst, Matrix a, Matrix b)
{
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.m[i] = a.m[i] + b.m[i];
    }
}

void matrix_sum(Matrix dst, Matrix b)
{
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.m[i] += b.m[i];
    }
}

void matrix_mult(Matrix dst, Matrix a, Matrix b)
{
    size_t i = 0, j = 0, k = 0;
    for (i = 0; i < dst.rows; ++i) {
        double* c = dst.m + i * dst.cols;

        for (j = 0; j < dst.cols; ++j)
            c[j] = 0;

        for (k = 0; k < b.rows; ++k) {
            const double* bb = b.m + k * dst.cols;
            double aa = a.m[i * a.cols + k];

            for (j = 0; j < dst.cols; ++j)
                c[j] += aa * bb[j];
        }
    }
}

void matrix_activation(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = sigmoid(MATRIX_AT(m, i, j));
        }
    }
}

void matrix_free(Matrix matrix)
{
    free(matrix.m);
}

void matrix_print(Matrix m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MATRIX_AT(m, i, j));
        }

        putchar('\n');
    }
}

#ifdef __cplusplus
}
#endif