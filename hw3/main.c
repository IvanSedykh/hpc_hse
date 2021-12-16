#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "mkl.h"
#include "mpi.h"
#include "omp.h"

typedef struct Matrix {
    double *data;
    int rows;
    int cols;
} Matrix;

void print_vector(const double *x, size_t N) {
    for (int i = 0; i < N; i++)
        printf("%4.3f ", x[i]);
    printf("\n");
}

void print_matrix(const Matrix *A) {
    for (int i = 0; i < A->rows; i++)
        print_vector(&(A->data[i * A->cols]), A->cols);
    printf("\n");
}

void init_matrix(Matrix *A, int NROWS, int NCOLS) {
    A->rows = NROWS;
    A->cols = NCOLS;
    A->data = calloc(sizeof(double), A->rows * A->cols);
}

// copy B <- A
void my_cblas_dcopy(const size_t N, const double *A, const int incA, double *B, const int incB) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        B[i] = A[i];
    }
}

// Y <- Y + alpha * X
void my_cblas_daxpy(const size_t N, const double alpha, const double *X,
                    const int incX, double *Y, const int incY) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        Y[i] += alpha * X[i];
    }
}

// X <- alpha * X
void my_cblas_dscal(const int N, const double alpha, double *X, const int incX) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        X[i] *= alpha;
    }
}

//  returns dot product X.Y
double my_cblas_ddot(const int N, const double *X, const int incX,
                     const double *Y, const int incY) {
    double res = 0.0;
#pragma omp parallel for reduction(+ \
                                   : res)
    for (size_t i = 0; i < N; i++) {
        res += X[i] * Y[i];
    }
    return res;
}

// matvec Y <- alpha * Ax + beta * Y
void my_cblas_dgemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const int M, const int N, const double alpha, const double *A,
                    const int lda, const double *X, const int incX,
                    const double beta, double *Y, const int incY) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        Y[i] *= beta;
        for (size_t j = 0; j < M; j++) {
            Y[i] += alpha * A[M * i + j] * X[j];
        }
    }
}

int main(int argc, char const *argv[]) {
    int debug = 0;
    int N = 1000;
    double eps = 1e-8;  // stopping criteria
    double err = 0.0;

    // __________MPI_______________
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        printf("Using %d procs\n", size);
    }

    // assume N divisible by size
    if (N % size != 0) {
        printf("N cannot be divided by nprocs\n");
        MPI_Abort(MPI_COMM_WORLD, 10);
    }
    // __________MPI_______________

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

    // split matrix row-wise
    int local_N = N / size;
    Matrix local_A;
    init_matrix(&local_A, local_N, N);
    printf("[%d] Local_A shape= (%d, %d)\n", rank, local_N, N);

    Matrix local_Az;
    init_matrix(&local_Az, 1, local_N);

    int tid;
#pragma omp parallel
    {
        tid = omp_get_thread_num();
        if (tid == 0) {
            int num_threads = omp_get_num_threads();
            printf("[%d] Using %d threads\n", rank, num_threads);
        }
    }

    // reading in 0 process
    if (rank == 0) {
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

        if (debug) {
            printf("root [%d] has A:\n", rank);
            print_matrix(&A);
        }
    }
    // initialize
    double alpha = 0.0;
    double beta = 0.0;
    double r_prev_dot_r_prev = 1.0;  // result of previous r.r
    if (rank == 0) {
        my_cblas_dscal(N, 1.0, x.data, 1);  // x <- 1.0

        my_cblas_dcopy(N, x.data, 1, r.data, 1);  // r <- x

        // r <- -1.0 Ax
        my_cblas_dgemv(CblasRowMajor, CblasNoTrans,
                       N, N, -1.0, A.data, N, x.data, 1, 0.0, r.data, 1);

        my_cblas_daxpy(N, 1.0, b.data, 1, r.data, 1);  // r <- b + r

        my_cblas_dcopy(N, r.data, 1, z.data, 1);  // z <- r
    }

    // _____________MPI_____________
    // send piece of matrix A to every process
    MPI_Scatter(
        A.data,
        local_N * N,
        MPI_DOUBLE,
        local_A.data,
        local_N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);
    // _____________MPI_____________
    if (debug) {
        printf(" proc [%d] has local_A:\n", rank);
        print_matrix(&local_A);
    }
    // ________MAIN LOOP ______________
    double start = dsecnd();
    for (size_t i = 0; i < 10 * N; i++) {
        if (rank == 0 && i % 50 == 0)
            printf("Iter:\t%d\tError:\t%f\n", i, err);
        if (debug && rank == 0) {
            printf("root proc [%d] has z:\n", rank);
            print_matrix(&z);
        }
        // _____________MPI_____________
        // send z to everyone
        MPI_Bcast(
            z.data,
            N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);
        // _____________MPI_____________
        if (debug) {
            printf("proc[%d] has z:\n", rank);
            print_matrix(&z);
        }

        // local_Az <- A_local*z
        my_cblas_dgemv(CblasRowMajor, CblasNoTrans,
                       N, local_N, 1.0, local_A.data, N, z.data, 1, 0.0, local_Az.data, 1);
        if (debug) {
            printf("[%d] has local_Az:\n", rank);
            print_matrix(&local_Az);
        }

        // _______MPI__________
        // gather Az from every proc
        MPI_Gather(
            local_Az.data,
            local_N,
            MPI_DOUBLE,
            Az.data,
            local_N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);
        // _______MPI__________
        if (debug && rank == 0) {
            printf("root [%d] has Az:\n", rank);
            print_matrix(&Az);
        }

        if (rank == 0) {
            alpha = my_cblas_ddot(N, r.data, 1, r.data, 1) /
                    my_cblas_ddot(N, Az.data, 1, z.data, 1);

            // if ||z|| < eps stop
            err = fabs(cblas_dnrm2(N, z.data, 1));
            if (err < eps)
            // if (fabs(cblas_dnrm2(N, r.data, 1)) < eps)
            {
                printf("Stopped after %d iterations\n", i);
                break;
            }

            my_cblas_daxpy(N, alpha, z.data, 1, x.data, 1);  // x <- x + alpha*z

            r_prev_dot_r_prev = my_cblas_ddot(N, r.data, 1, r.data, 1);  // r.r <- r.r

            my_cblas_daxpy(N, -alpha, Az.data, 1, r.data, 1);  // r <- r - alpha * Az

            beta = my_cblas_ddot(N, r.data, 1, r.data, 1) / r_prev_dot_r_prev;

            my_cblas_dscal(N, beta, z.data, 1);  // z <- beta*z
            my_cblas_daxpy(N, 1.0, r.data, 1, z.data, 1);

            // printf("i=%d\n", i);
            // print_matrix(&x);
        }
    }

    if (rank == 0) {
        double total_time = (dsecnd() - start) * 1000;  // total time in ms
        printf("Done in %.5f milliseconds \n", total_time);
    }

    // printf("result x:\n");
    // print_matrix(&x);

    if (rank == 0) {
        // write output to file
        FILE *out_file = fopen("output.txt", "w");
        if (out_file == NULL)
            printf("Bad output file\n");
        for (size_t i = 0; i < N; i++)
            fprintf(out_file, "%.12f\n", x.data[i]);
        fprintf(out_file, "\n");
        fclose(out_file);
    }

    // deallocating memory
    free(A.data);
    free(b.data);
    free(x.data);
    free(r.data);
    free(z.data);

    // ______MPI________
    MPI_Finalize();
    // ______MPI________
    return 0;
}
