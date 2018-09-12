
/*
 *  Code based on the example by Jesse Laning
 */

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "CL/cl2.hpp"

/** Load the kernel in from the file name into a string object */
static std::string load_kernel_source(std::string const name)
{
    std::ifstream in(name.c_str());
    std::string source_code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return source_code;
}

static bool is_valid(cl::Device const& device)
{
    return device.getInfo<CL_DEVICE_AVAILABLE>();
    //&& device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
}

// Need to find the best device according to:
// 1) Number of compute units (highest)
// 2) Amount of memory
// 3) Availability
// 4) Double floating point precision availability
void remove_invalid(std::vector<cl::Platform>& platforms)
{
    platforms.erase(std::remove_if(begin(platforms),
                                   end(platforms),
                                   [](cl::Platform const& platform) {
                                       // Check each device for this platform
                                       cl::vector<cl::Device> devices;
                                       platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU,
                                                           &devices);

                                       return devices.empty()
                                              || std::none_of(begin(devices),
                                                              end(devices),
                                                              [](cl::Device const& device) {
                                                                  return is_valid(device);
                                                              });
                                   }),
                    end(platforms));
}

void remove_invalid(std::vector<cl::Device>& devices)
{
    devices.erase(std::remove_if(begin(devices),
                                 end(devices),
                                 [](cl::Device const& device) { return !is_valid(device); }),
                  end(devices));
}

struct highest_compute
{
    auto operator()(cl::Device const& left, cl::Device const& right) const noexcept -> bool
    {
        return left.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
               < right.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    }
};

struct highest_memory
{
    auto operator()(cl::Device const& left, cl::Device const& right) const noexcept -> bool
    {
        return left.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() < right.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    }
};

int main(int const argc, char* argv[])
{
    std::vector<float> x(100'000'000, 1.0f), y(100'000'000, 1.0f);

    // Populate with the available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        throw std::domain_error("No OpenCL platforms found");
    }

    std::cout << "Number of raw OpenCL platforms: " << platforms.size() << "\n";
    remove_invalid(platforms);
    std::cout << "Filtered number of platforms: " << platforms.size() << '\n';

    // std::map<cl::Platform, std::vector<cl::Device>> platform_devices;

    cl::Device device;

    for (auto const& platform : platforms)
    {
        std::string output;
        platform.getInfo(CL_PLATFORM_NAME, &output);

        std::cout << "output: " << output << "\n";

        // Check each device for this platform
        cl::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        if (devices.empty())
        {
            std::cout << "no devices for platform, skipping\n";
            continue;
        }

        remove_invalid(devices);

        device = *std::max_element(begin(devices), end(devices), highest_compute{});

        std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';

        std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
        std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")\n";

        std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << '\n';

        std::cout << "\t\tDevice Max Compute Units: "
                  << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << '\n';

        std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                  << '\n';

        std::cout << "\t\tDevice Max Clock Frequency: "
                  << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << '\n';

        std::cout << "\t\tDevice Max Allocatable Memory: "
                  << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << '\n';

        std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                  << '\n';

        std::cout << "\t\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << '\n';
    }

    // Set the device to the first device in the platform.
    // You can have more than one device associated with a single platform,
    // for instance if you had two of the same GPUs on your system in SLI or CrossFire.
    // cl::Device device = devices.at(0);

    // This is just helpful to see what device and platform you are using.
    // std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    // std::cout << "Using platform: " << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::cout << "creating the opencl context\n";

    // Finally create the OpenCL context from the device you have chosen.
    cl::Context context(device);

    cl::Buffer ocl_x(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
    cl::Buffer ocl_y(context, CL_MEM_READ_WRITE, sizeof(float) * y.size());

    // A source object for your program
    std::string kernel_code = load_kernel_source("../kernels/saxpy.cl");

    // Add your program source
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // Create your OpenCL program and build it.
    cl::Program program(context, sources);

    if (program.build({device}) != CL_SUCCESS)
    {
        // Print the build log to find any issues with your source
        throw std::domain_error("Error building program: "
                                + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    }

    // Create our kernel where the "<>" is the name of the OpenCL kernel
    cl::Kernel saxpy(program, "saxpy");

    // Set our arguments for the kernel (x, y, a)
    saxpy.setArg(0, ocl_x);
    saxpy.setArg(1, ocl_y);
    saxpy.setArg(2, 2.0f);

    cl::CommandQueue queue(context, device, 0, nullptr);

    // Write our buffers that we are adding to our OpenCL device
    queue.enqueueWriteBuffer(ocl_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
    queue.enqueueWriteBuffer(ocl_y, CL_TRUE, 0, sizeof(float) * y.size(), y.data());
    queue.finish();

    // Create an event that we can use to wait for our program to finish running
    cl::Event event;

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
    queue.enqueueReadBuffer(ocl_y, CL_TRUE, 0, sizeof(float) * x.size(), y.data());
    queue.finish();

    // Prints the array
    if (std::any_of(begin(y), end(y), [](auto const i) { return i != 3; }))
    {
        throw std::domain_error("saxpy computation was not successful");
    }
    std::cout << "\nComputation successful\n\n";

    return 0;
}
