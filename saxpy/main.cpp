/*
 *  To compile use
 *  g++ -std=c++11 main.cpp -lOpenCL
 *
 *  Code based on the example by Jesse Laning
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
    #include "CL/cl.hpp"
#endif

/** Load the kernel in from the file name into a string object */
std::string LoadKernel(const char* name)
{
	std::ifstream in(name);
	std::string sourceCode((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	return sourceCode;
}

int main()
{
    const int size = 10;
    float x[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    float y[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    // Populate with the available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0)
    {
    	std::cout << "No OpenCL platforms found" << std::endl;
    	exit(1);
    }

    std::cout << "The number of platforms is " << platforms.size() << "\n";

    // Create a stl vector to store all of the availbe devices to use from the first platform.
    std::vector<cl::Device> devices;

    // TODO Use some method of getting the best OpenCL device
    const int platformId = 0;

    // Get the available devices from the platform.
    platforms[platformId].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // Set the device to the first device in the platform.
    // You can have more than one device associated with a single platform,
    // for instance if you had two of the same GPUs on your system in SLI or CrossFire.
    cl::Device device = devices[0];

    // This is just helpful to see what device and platform you are using.
    std::cout << "Using device: "   << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Using platform: " << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << std::endl;

    // Finally create the OpenCL context from the device you have chosen.
    cl::Context context(device);

    cl::Buffer ocl_x(context, CL_MEM_READ_WRITE, sizeof(float) * size);
    cl::Buffer ocl_y(context, CL_MEM_READ_WRITE, sizeof(float) * size);

    // A source object for your program
    cl::Program::Sources sources;
    std::string kernel_code = LoadKernel("kernels/saxpy.cl");

    // Add your program source
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // Create your OpenCL program and build it.
    cl::Program program(context, sources);

    if (program.build({device}) != CL_SUCCESS)
    {
        // Print the build log to find any issues with your source
    	std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    	return 1;
    }

    // Create our kernel where the "<>" is the name of the OpenCL kernel
    cl::Kernel saxpy(program, "saxpy");

    // Set our arguments for the kernel (x, y, a)
    saxpy.setArg(0, ocl_x);
    saxpy.setArg(1, ocl_y);
    saxpy.setArg(2, 2.0f);

    cl::CommandQueue queue(context, device, 0, nullptr);

    // Write our buffers that we are adding to our OpenCL device
    queue.enqueueWriteBuffer(ocl_x, CL_TRUE, 0, sizeof(float) * size, x);
    queue.enqueueWriteBuffer(ocl_y, CL_TRUE, 0, sizeof(float) * size, y);
    queue.finish();

    // Create an event that we can use to wait for our program to finish running
    cl::Event event;

    // This runs our program, the ranges here are the offset, global, local ranges that our code runs in.
    queue.enqueueNDRangeKernel( saxpy,
                                cl::NullRange,     // Offset
                                cl::NDRange(size), // Global size
                                cl::NullRange,     // Local offset
                                0,                 // Local size
                                &event);
    event.wait();

    // Reads the output written to our buffer into our final array
    queue.enqueueReadBuffer(ocl_y, CL_TRUE, 0, sizeof(float) * size, y);
    queue.finish();

    // Prints the array
    std::cout << "Result for saxpy kernel\n";
    for (int i = 0; i < size; i++)
    {
    	std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    return 0;
}
