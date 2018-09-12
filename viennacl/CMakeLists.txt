cmake_minimum_required(VERSION 3.0)
project (hello)

include_directories ("/usr/include/eigen3/")
add_executable(hello viennacl_svd.cpp)
target_link_libraries (hello OpenCL) 

# add_executable(viennacl_svd.cpp)
# target_link_libraries(opencl)
