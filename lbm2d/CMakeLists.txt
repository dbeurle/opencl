
cmake_minimum_required(VERSION 3.9)

project(lattice_boltzmann LANGUAGES CUDA CXX)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL COMPONENTS OpenGL REQUIRED)

add_executable(lb 2dLB.cu)

target_compile_features(lb PUBLIC cxx_std_11)
target_link_libraries(lb GLUT::GLUT GLEW::GLEW OpenGL::GL)
