
cmake_minimum_required(VERSION 3.9)

project(svd_test LANGUAGES CUDA CXX)

find_package(CUDA)

add_executable(svd_qr_test svd_qr.cu)
target_link_libraries(svd_qr_test PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

add_executable(svd_jacobi_test svd_jacobi.cu)
target_link_libraries(svd_jacobi_test PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
