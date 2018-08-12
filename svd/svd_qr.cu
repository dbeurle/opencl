
// C++-ified SVD example from
// https://docs.nvidia.com/cuda/cusolver/index.html#svd_examples

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>
#include <iostream>
#include <vector>

inline void printMatrix(int m, int n, const double* A, int lda, const char* name)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            std::cout << name << "(" << row + 1 << ", " << col + 1 << ") = " << A[row + col * lda]
                      << '\n';
        }
    }
}

inline void check(cudaError_t&& error)
{
    if (error != cudaSuccess)
    {
        throw std::domain_error("CUDA error");
    }
}

inline void check(cublasStatus_t&& status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::domain_error("Cuda BLAS error");
    }
}

inline void check(cusolverStatus_t&& status)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        throw std::domain_error("Cuda BLAS error");
    }
}

int main(int argc, char* argv[])
{
    constexpr int rows = 3;
    constexpr int cols = 2;
    constexpr int lda = rows;

    //       | 1 2  |
    //   A = | 4 5  |
    //       | 2 1  |
    std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};

    int info_gpu = 0;

    std::cout << "A = (matlab base-1)\n";
    printMatrix(rows, cols, A.data(), lda, "A");
    std::cout << "=====\n";

    // step 1: create cusolverDn/cublas handle
    cusolverDnHandle_t cusolver_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    check(cusolverDnCreate(&cusolver_handle));
    check(cublasCreate(&cublas_handle));

    // step 2: copy A and B to device
    double* d_A = nullptr;
    double* d_S = nullptr;
    double* d_U = nullptr;
    double* d_VT = nullptr;
    int* devInfo = nullptr;
    double* d_W = nullptr; // W = S*VT

    check(cudaMalloc((void**)&d_A, sizeof(double) * lda * cols));
    check(cudaMalloc((void**)&d_S, sizeof(double) * cols));
    check(cudaMalloc((void**)&d_U, sizeof(double) * lda * rows));
    check(cudaMalloc((void**)&d_VT, sizeof(double) * lda * cols));
    check(cudaMalloc((void**)&devInfo, sizeof(int)));
    check(cudaMalloc((void**)&d_W, sizeof(double) * lda * cols));

    check(cudaMemcpy(d_A, A.data(), sizeof(double) * lda * cols, cudaMemcpyHostToDevice));

    // step 3: query working space of SVD
    int lwork = 0;
    check(cusolverDnDgesvd_bufferSize(cusolver_handle, rows, cols, &lwork));

    double* d_work = nullptr;
    check(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    // step 4: compute SVD
    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT

    double* d_rwork = nullptr;

    check(cusolverDnDgesvd(cusolver_handle,
                           jobu,
                           jobvt,
                           rows,
                           cols,
                           d_A,
                           lda,
                           d_S,
                           d_U,
                           lda, // ldu
                           d_VT,
                           lda, // ldvt,
                           d_work,
                           lwork,
                           d_rwork,
                           devInfo));

    check(cudaDeviceSynchronize());

    // m-by-m unitary matrix
    std::vector<double> U(lda * rows);
    // n-by-n unitary matrix
    std::vector<double> VT(lda * cols);
    // singular value
    std::vector<double> S(cols);

    check(cudaMemcpy(U.data(), d_U, sizeof(double) * lda * rows, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(VT.data(), d_VT, sizeof(double) * lda * cols, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(S.data(), d_S, sizeof(double) * cols, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "after gesvd: info_gpu = " << info_gpu << "\n";

    assert(info_gpu == 0);

    std::cout << "=====\n";

    std::cout << "S = (matlab base-1)\n";
    printMatrix(cols, 1, S.data(), lda, "S");
    std::cout << "=====\n";

    std::cout << "U = (matlab base-1)\n";
    printMatrix(rows, rows, U.data(), lda, "U");
    std::cout << "=====\n";

    std::cout << "VT = (matlab base-1)\n";
    printMatrix(cols, cols, VT.data(), lda, "VT");
    std::cout << "=====\n";

    // step 5: measure error of singular value
    std::vector<double> const S_exact = {7.065283497082729, 1.040081297712078};

    double ds_sup = 0.0;
    for (int j = 0; j < cols; j++)
    {
        ds_sup = std::max(ds_sup, std::abs(S[j] - S_exact[j]));
    }

    std::cout << "|S - S_exact| = " << ds_sup << "\n";

    // step 6: |A - U*S*VT|
    // W = S*VT
    check(cublasDdgmm(cublas_handle, CUBLAS_SIDE_LEFT, cols, cols, d_VT, lda, d_S, 1, d_W, lda));

    // A := -U*W + A
    check(cudaMemcpy(d_A, A.data(), sizeof(double) * lda * cols, cudaMemcpyHostToDevice));

    double constexpr h_one = 1;
    double constexpr h_minus_one = -1;

    check(cublasDgemm_v2(cublas_handle,
                         CUBLAS_OP_N,  // U
                         CUBLAS_OP_N,  // W
                         rows,         // number of rows of A
                         cols,         // number of columns of A
                         cols,         // number of columns of U
                         &h_minus_one, // host pointer
                         d_U,          // U
                         lda,          // Leading dimension
                         d_W,          // W
                         lda,          // Leading dimension
                         &h_one,       // hostpointer
                         d_A,          // Device matrix
                         lda));

    double dR_fro = 0.0;
    check(cublasDnrm2_v2(cublas_handle, lda * cols, d_A, 1, &dR_fro));

    std::cout << "|A - U*S*VT| = " << dR_fro << "\n";

    // Deallocate memory
    if (d_A) cudaFree(d_A);
    if (d_S) cudaFree(d_S);
    if (d_U) cudaFree(d_U);
    if (d_VT) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    if (d_W) cudaFree(d_W);

    // Destroy the handles
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cusolver_handle) cusolverDnDestroy(cusolver_handle);

    cudaDeviceReset();

    return 0;
}
