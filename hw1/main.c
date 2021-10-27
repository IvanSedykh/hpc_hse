#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include "mkl.h"

typedef struct Matrix
{
    double *data;
    int rows;
    int cols;
} Matrix;

void print_vector(const double *x, size_t N)
{
    for (int i = 0; i < N; i++)
        printf("%4.8f ", x[i]);
    printf("\n");
}

void print_matrix(const Matrix *A)
{
    for (int i = 0; i < A->rows; i++)
        print_vector(&(A->data[i * A->cols]), A->cols);
    printf("\n");
}

void init_matrix(Matrix *A, int NROWS, int NCOLS)
{
    A->rows = NROWS;
    A->cols = NCOLS;
    A->data = calloc(sizeof(double), A->rows * A->cols);
}

int main(int argc, char const *argv[])
{
    int N = 10;
    double eps = 1e-8; // stopping criteria

    Matrix A;
    init_matrix(&A, N, N);

    Matrix b;
    init_matrix(&b, 1, N);

    Matrix x;
    init_matrix(&x, 1, N);

    Matrix r;
    init_matrix(&r, 1, N);

    Matrix z;
    init_matrix(&z, 1, N);

    Matrix Az;
    init_matrix(&Az, 1, N);

    // read input from file
    FILE *in_file = fopen("input.txt", "r");
    if (in_file == NULL)
        printf("Bad input file\n");
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            fscanf(in_file, "%lf", &(A.data[i * A.cols + j]));
    for (size_t i = 0; i < N; i++)
        fscanf(in_file, "%lf", &(b.data[i]));
    fclose(in_file);

    // initialize
    double alpha = 0.0;
    double beta = 0.0;
    double r_prev_dot_r_prev = 1.0;       // result of previous r.r
    cblas_dscal(N, 1.0, x.data, 1);       // x <- 1.0
    cblas_dcopy(N, x.data, 1, r.data, 1); // r <- x
    // r <- -1.0 Ax
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                N, N, -1.0, A.data, N, x.data, 1, 0.0, r.data, 1);
    cblas_daxpy(N, 1.0, b.data, 1, r.data, 1); // r <- b + r
    cblas_dcopy(N, r.data, 1, z.data, 1);      // z <- r

    for (size_t i = 0; i < 2 * N; i++)
    {
        // Az <- A*z
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    N, N, 1.0, A.data, N, z.data, 1, 0.0, Az.data, 1);

        alpha = cblas_ddot(N, r.data, 1, r.data, 1) /
                cblas_ddot(N, Az.data, 1, z.data, 1);

        // if ||z|| < eps stop
        if (fabs(cblas_dnrm2(N, z.data, 1)) < eps)
        {
            printf("Stopped after %d iterations\n", i);
            break;
        }

        cblas_daxpy(N, alpha, z.data, 1, x.data, 1); // x <- x + alpha*z

        r_prev_dot_r_prev = cblas_ddot(N, r.data, 1, r.data, 1); // r.r <- r.r

        cblas_daxpy(N, -alpha, Az.data, 1, r.data, 1); // r <- r - alpha * Az

        beta = cblas_ddot(N, r.data, 1, r.data, 1) / r_prev_dot_r_prev;

        cblas_dscal(N, beta, z.data, 1); // z <- beta*z
        cblas_daxpy(N, 1.0, r.data, 1, z.data, 1);

        printf("i=%d\n", i);
        print_matrix(&x);
    }

    printf("result x:\n");
    print_matrix(&x);

    // write output to file
    FILE *out_file = fopen("output.txt", "w");
    if (out_file == NULL)
        printf("Bad output file\n");
    for (size_t i = 0; i < N; i++)
        fprintf(out_file, "%.8f ", x.data[i]);
    fprintf(out_file, "\n");
    fclose(out_file);

    // deallocating memory
    free(A.data);
    free(b.data);
    free(x.data);
    free(r.data);
    free(z.data);

    return 0;
}
