#include <cosma/local_multiply.hpp>
#include <cosma/multiply.hpp>

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>
#include <mpi.h>

#include <cmath>
#include <limits>

template <typename scalar>
void fill_matrix(cosma::CosmaMatrix<scalar> &M) {
    for (int idx = 0; idx < M.matrix_size(); ++idx) {
        M.matrix_pointer()[idx] = std::sin(idx);
    }
};

// !!! [NOTE] The test depends on correct implementation of `multiply()`.
//
TEST(MultiplyUsingLayout, ) {
    using scalar_t = double;
    constexpr int nprocs = 4;
    constexpr int m = 20;
    constexpr int n = 20;
    constexpr int k = 80;
    constexpr scalar_t alpha = 1;
    constexpr scalar_t beta = 1;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    cosma::Strategy strategy(m, n, k, nprocs);

    // create a separate context
    auto ctx = cosma::make_context<scalar_t>();

    // these matrices have to be created within a context
    // that is separated from the context that is used
    // in multiply_using_layout, because the memory pool
    // might resize within multiply_using_layout
    // and the A_grid pointers might become outdated
    cosma::CosmaMatrix<scalar_t> A(ctx, 'A', strategy, rank);
    cosma::CosmaMatrix<scalar_t> B(ctx, 'B', strategy, rank);
    cosma::CosmaMatrix<scalar_t> C(ctx, 'C', strategy, rank);

    fill_matrix(A);
    fill_matrix(B);
    fill_matrix(C);

    // important if beta > 0
    cosma::CosmaMatrix<scalar_t> C_act('C', strategy, rank);
    for (int idx = 0; idx < C_act.matrix_size(); ++idx) {
        C_act.matrix_pointer()[idx] = C.matrix_pointer()[idx];
    }

    auto A_grid = A.get_grid_layout();
    auto B_grid = B.get_grid_layout();
    auto C_grid = C.get_grid_layout();

    // This routine should not generally be used for COSMA matrices.
    // If it is used, then we must ensure the context of matrices A_grid, B_grid and C_grid
    // is not the default context (singleton) that multiply_using_layout is using
    // because the pointers A_grid, B_grid and C_grid must be persistent 
    // throughout the whole execution of multiply_using_layout
    cosma::multiply_using_layout(A_grid, B_grid, C_grid, alpha, beta, comm);

    cosma::multiply(ctx, A, B, C_act, strategy, comm, alpha, beta);

    // ----- Checks for data integrity

    scalar_t *C_data = C.matrix_pointer();
    scalar_t *C_act_data = C_act.matrix_pointer();

    // Check if sizes match.
    //
    ASSERT_EQ(C.matrix_size(), C_act.matrix_size());

    // Check if data elements match. Fail if an element doesn't match.
    //
    for (int i = 0; i < C_act.matrix_size(); ++i) {
        ASSERT_DOUBLE_EQ(C_data[i], C_act_data[i]);
    }
}
