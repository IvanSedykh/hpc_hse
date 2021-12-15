#include "mpi.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "mkl.h"
#include "omp.h"

int main(int argc, char const *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("proc [%d] of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
