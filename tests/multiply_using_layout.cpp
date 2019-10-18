#include <cosma/local_multiply.hpp>
#include <cosma/multiply.hpp>

#include <mpi.h>

// Returns the number of processes at runtime
//
int num_procs(MPI_Comm comm) {
    int num_procs;
    MPI_Comm_size(comm, &num_procs);
    return num_procs;
}

// The tests simulates a tall sckinny QR
//
int main() {
    using scalar_t = double;
    constexpr int nprocs = 4;
    constexpr int m = 20;
    constexpr int n = 20;
    constexpr int k = 80;
    constexpr scalar_t alpha = 1;
    constexpr scalar_t beta = 0;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (nprocs != num_procs(comm)) {
        printf("The test runs with %d processes!", nprocs);
        MPI_Abort(comm, 1);
    }

    cosma::Strategy strategy(m, n, k, nprocs);

    cosma::CosmaMatrix<scalar_t> A('A', strategy, rank);
    cosma::CosmaMatrix<scalar_t> B('B', strategy, rank);
    cosma::CosmaMatrix<scalar_t> C('C', strategy, rank);

    auto fill_matrix = [](cosma::CosmaMatrix<scalar_t> &M) {
        for (int idx = 0; idx < M.matrix_size(); ++idx) {
            M[idx] = idx;
        }
    };

    fill_matrix(A);
    fill_matrix(B);

    auto A_grid = A.get_grid_layout();
    auto B_grid = B.get_grid_layout();
    auto C_grid = C.get_grid_layout();
    cosma::multiply_using_layout(A_grid, B_grid, C_grid, alpha, beta, comm);

    cosma::CosmaMatrix<scalar_t> C_act('C', strategy, rank);
}
