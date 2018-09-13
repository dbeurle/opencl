
FROM fedora:28

RUN dnf update -y
# Compilers and build system
RUN dnf install -y wget cmake clang clang-tools-extra libomp-devel gcc-c++

# OpenCL runtimes
RUN dnf install -y ocl-icd clpeak clinfo
# Intel Xeon OpenCL runtime
RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz && tar -xf opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz
COPY docker/silent.cfg /opencl_runtime_16.1.2_x64_rh_6.4.0.37
RUN cd opencl_runtime_16.1.2_x64_rh_6.4.0.37 && sh install.sh --silent silent.cfg
