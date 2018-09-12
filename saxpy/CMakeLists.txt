cmake_minimum_required(VERSION 3.8)

project(opencl_test CXX)

set(CMAKE_CXX_STANDARD 14)
add_definitions("-Wall")

option(ENABLE_NATIVE "Optimise for current architecture" OFF)

if (ENABLE_NATIVE)
    add_definitions("-march=native -mtune=native")
endif()

find_package(OpenCL REQUIRED)

add_executable(saxpy main.cpp)

target_link_libraries(saxpy OpenCL)
