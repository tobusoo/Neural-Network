#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        double *m;
        size_t rows;
        size_t cols;
    } Matrix;

#define MATRIX_AT(matrix, i, j) ((matrix).m[(i) * (matrix).cols + (j)])

    Matrix matrix_alloc(size_t rows, size_t cols);
    void matrix_rand(Matrix m, double low, double high);
    void matrix_fill(Matrix m, double x);

    void matrix_sum(Matrix dst, Matrix b);
    void matrix_sum_dst(Matrix dst, Matrix a, Matrix b);
    void matrix_mult(Matrix dst, Matrix a, Matrix b);

    void matrix_activation(Matrix m);
    void matrix_print(Matrix m);
    void matrix_free(Matrix m);
#ifdef __cplusplus
}
#endif

#endif // MATRIX_H
