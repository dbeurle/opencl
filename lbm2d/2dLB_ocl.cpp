
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#define USE_NVIDIA_DEVICE

#ifdef USE_NVIDIA_DEVICE
// Doesn't work with nvidia backend otherwise
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#include "CL/cl2.hpp"

/// Load the kernel in from the file name into a string object
inline std::string load_kernel_source(std::string&& name)
{
    std::ifstream in(name.c_str());
    std::string source_code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return source_code;
}

int main(int const argc, char* argv[])
{
    // Populate with the available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        throw std::domain_error("No OpenCL platforms found");
    }

    std::cout << "Number of platforms: " << platforms.size() << "\n";

    {
        std::int32_t platform_counter{0};

        for (auto const& platform : platforms)
        {
            std::string output;
            platform.getInfo(CL_PLATFORM_NAME, &output);

            std::cout << "Platform " << platform_counter << ": " << output << std::endl;

            platform_counter++;
        }
    }

    // TODO Use some method of getting the best OpenCL device
    const int platformId = 0;

    std::cout << "Using platform " << platformId << "\n";

    // Create a stl vector to store all of the availbe devices to use from the first
    // platform.
    std::vector<cl::Device> devices;

    // Get the available devices from the platform.
    platforms[platformId].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    std::cout << "Number of devices found: " << devices.size() << "\n";

    // Set the device to the first device in the platform.
    // You can have more than one device associated with a single platform,
    // for instance if you had two of the same GPUs on your system in SLI or CrossFire.
    cl::Device device = devices.at(0);

    // This is just helpful to see what device and platform you are using.
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Using platform: " << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << std::endl;

    // Finally create the OpenCL context from the device you have chosen.
    cl::Context context(device);

    std::vector<float> x(100'000, 1.0f), y(100'000, 1.0f);

    cl::Buffer device_x(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
    cl::Buffer device_y(context, CL_MEM_READ_WRITE, sizeof(float) * y.size());

    // A source object for your program
    std::string kernel_code = load_kernel_source("../ocl/kernels/saxpy.cl");

    // Add your program source
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // Create your OpenCL program and build it.
    cl::Program program(context, sources);

    if (program.build({device}) != CL_SUCCESS)
    {
        // Print the build log to find any issues with your source
        throw std::domain_error("Error building: "
                                + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    }

    // Create our kernel where the "<>" is the name of the OpenCL kernel
    cl::Kernel saxpy(program, "saxpy");

    // Set our arguments for the kernel (x, y, a)
    saxpy.setArg(0, device_x);
    saxpy.setArg(1, device_y);
    saxpy.setArg(2, 2.0f);

    cl::CommandQueue queue(context, device, 0, nullptr);

    // Write our buffers that we are adding to our OpenCL device
    queue.enqueueWriteBuffer(device_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
    queue.enqueueWriteBuffer(device_y, CL_TRUE, 0, sizeof(float) * y.size(), y.data());

    // Create an event that we can use to wait for our program to finish running
    cl::Event event;

    queue.finish();

    // This runs our program, the ranges here are the offset, global, local ranges that
    // our code runs in.
    queue.enqueueNDRangeKernel(saxpy,
                               cl::NullRange,         // Offset
                               cl::NDRange(x.size()), // Global size
                               cl::NullRange,         // Local offset
                               NULL,                  // Local size
                               &event);
    event.wait();

    // Reads the output written to our buffer into our final array
    queue.enqueueReadBuffer(device_y, CL_TRUE, 0, sizeof(decltype(x)::value_type) * x.size(), y.data());
    queue.finish();

    // A very dubuous check
    if (std::none_of(begin(y), end(y), [](auto const i) { return i == 3.0f; }))
    {
        throw std::domain_error("saxpy computation was not successful");
    }
    std::cout << "\nComputation successful\n\n";

    return 0;
}
