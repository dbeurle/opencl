
cmake_minimum_required(VERSION 3.9)

project(lattice_boltzmann LANGUAGES CUDA CXX)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL COMPONENTS OpenGL REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(lbm_cuda 2dLB.cu)
add_executable(lbm_cpp 2dLB.cpp)
add_executable(lbm_ocl 2dLB_ocl.cpp)

target_compile_features(lbm_cuda PUBLIC cxx_std_11)
target_compile_features(lbm_cpp PUBLIC cxx_std_11)
target_compile_features(lbm_ocl PUBLIC cxx_std_14)

target_link_libraries(lbm_cuda GLUT::GLUT GLEW::GLEW OpenGL::GL)
target_link_libraries(lbm_cpp GLUT::GLUT GLEW::GLEW OpenGL::GL)
target_link_libraries(lbm_ocl OpenCL::OpenCL)
