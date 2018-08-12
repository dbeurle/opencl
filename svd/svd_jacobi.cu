
// C++-ified SVD example from
// https://docs.nvidia.com/cuda/cusolver/index.html#svd_examples

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cassert>
#include <vector>
#include <iostream>

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
    cusolverDnHandle_t cusolverH = nullptr;
    cudaStream_t stream = nullptr;
    gesvdjInfo_t gesvdj_params = nullptr;

    const int m = 3;
    const int n = 2;
    const int lda = m;

    ///       | 1 2  |
    ///   A = | 4 5  |
    ///       | 2 1  |
    std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    // m-by-m unitary matrix, left singular vectors
    std::vector<double> U(lda * m);
    // n-by-n unitary matrix, right singular vectors
    std::vector<double> V(lda * n);
    // numerical singular value
    std::vector<double> S(n);
    // exact singular values
    std::vector<double> const S_exact = {7.065283497082729, 1.040081297712078};

    // device copy of A
    double* d_A = nullptr;
    // singular values
    double* d_S = nullptr;
    // left singular vectors
    double* d_U = nullptr;
    // right singular vectors
    double* d_V = nullptr;
    // error info
    int* d_info = nullptr;

    // devie workspace for gesvdj
    double* d_work = nullptr;
    // host copy of error info
    int info = 0;

    // configuration of gesvdj
    constexpr double tol = 1.e-7;
    constexpr int max_sweeps = 15;

    // compute eigenvectors
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    // econ = 1 for economy size
    constexpr int econ = 0;

    // numerical results of gesvdj
    double residual = 0.0;

    int executed_sweeps = 0;

    std::cout << "example of gesvdj \n";
    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    printf("econ = %d \n", econ);

    std::cout << "A = (matlab base-1)\n";
    printMatrix(m, n, A.data(), lda, "A");
    std::cout << "=====\n";

    // step 1: create cusolver handle, bind a stream
    check(cusolverDnCreate(&cusolverH));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cusolverDnSetStream(cusolverH, stream));

    // step 2: configuration of gesvdj
    check(cusolverDnCreateGesvdjInfo(&gesvdj_params));
    // default value of tolerance is machine zero
    check(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
    // default value of max. sweeps is 100
    check(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

    // step 3: copy A and B to device
    check(cudaMalloc((void**)&d_A, sizeof(double) * lda * n));
    check(cudaMalloc((void**)&d_S, sizeof(double) * n));
    check(cudaMalloc((void**)&d_U, sizeof(double) * lda * m));
    check(cudaMalloc((void**)&d_V, sizeof(double) * lda * n));
    check(cudaMalloc((void**)&d_info, sizeof(int)));

    check(cudaMemcpy(d_A, A.data(), sizeof(double) * lda * n, cudaMemcpyHostToDevice));

    // step 4: query workspace of SVD

    // size of workspace
    int lwork = 0;
    check(
        cusolverDnDgesvdj_bufferSize(cusolverH,
                                     jobz, // CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only
                                     // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
                                     econ, // econ = 1 for economy size
                                     m,    // nubmer of rows of A, 0 <= m
                                     n,    // number of columns of A, 0 <= n
                                     d_A,  // m-by-n
                                     lda,  // leading dimension of A
                                     d_S,  // min(m,n)
                                           // the singular values in descending order
                                     d_U,  // m-by-m if econ = 0
                                           // m-by-min(m,n) if econ = 1
                                     lda,  // leading dimension of U, ldu >= max(1,m)
                                     d_V,  // n-by-n if econ = 0
                                           // n-by-min(m,n) if econ = 1
                                     lda,  // leading dimension of V, ldv >= max(1,n)
                                     &lwork,
                                     gesvdj_params));

    check(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    // step 5: compute SVD
    check(cusolverDnDgesvdj(cusolverH,
                            jobz, // CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only
                            // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
                            econ, // econ = 1 for economy size
                            m,    // nubmer of rows of A, 0 <= m
                            n,    // number of columns of A, 0 <= n
                            d_A,  // m-by-n
                            lda,  // leading dimension of A
                            d_S,  // min(m,n)
                                  // the singular values in descending order
                            d_U,  // m-by-m if econ = 0
                                  // m-by-min(m,n) if econ = 1
                            lda,  // leading dimension of U, ldu >= max(1,m)
                            d_V,  // n-by-n if econ = 0
                                  // n-by-min(m,n) if econ = 1
                            lda,  // leading dimension of V, ldv >= max(1,n)
                            d_work,
                            lwork,
                            d_info,
                            gesvdj_params));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(U.data(), d_U, sizeof(double) * lda * m, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(V.data(), d_V, sizeof(double) * lda * n, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(S.data(), d_S, sizeof(double) * n, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    check(cudaDeviceSynchronize());

    if (info == 0)
    {
        std::cout << "gesvdj converges\n";
    }
    else if (0 > info)
    {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    else
    {
        printf("WARNING: info = %d : gesvdj did not converge \n", info);
    }

    std::cout << "S = singular values (matlab base-1)\n";
    printMatrix(n, 1, S.data(), lda, "S");
    std::cout << "=====\n";

    std::cout << "U = left singular vectors (matlab base-1)\n";
    printMatrix(m, m, U.data(), lda, "U");
    std::cout << "=====\n";

    std::cout << "V = right singular vectors (matlab base-1)\n";
    printMatrix(n, n, V.data(), lda, "V");
    std::cout << "=====\n";

    /* step 6: measure error of singular value */
    double ds_sup = 0.0;
    for (int j = 0; j < n; j++)
    {
        ds_sup = std::max(ds_sup, std::abs(S[j] - S_exact[j]));
    }
    printf("|S - S_exact|_sup = %E \n", ds_sup);

    check(cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));

    check(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));

    printf("residual |A - U*S*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);

    if (d_A) cudaFree(d_A);
    if (d_S) cudaFree(d_S);
    if (d_U) cudaFree(d_U);
    if (d_V) cudaFree(d_V);
    if (d_info) cudaFree(d_info);
    if (d_work) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    cudaDeviceReset();

    return 0;
}
