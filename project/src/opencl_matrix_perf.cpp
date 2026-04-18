#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <cmath>

#include "matrix_opencl.hpp"

#define NUM_TESTS 10

cl::Context context;
cl::CommandQueue queue;

void setupOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);

    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    MatrixCL::initializeKernels(context, {device});
}

int main()
{
    setupOpenCL();

    std::filesystem::create_directories("performance");

#if CL_MUL_METHOD == 0
    const std::string method = "opencl_naive";
#elif CL_MUL_METHOD == 1
    #if TILE_SIZE == 4
        const std::string method = "opencl_tiled_4";
    #elif TILE_SIZE == 8
        const std::string method = "opencl_tiled_8";
    #else
        const std::string method = "opencl_tiled_16";
    #endif
#else
    exit(1);
#endif

    std::ofstream csv("performance/" + method + ".csv", std::ios::trunc);
    csv << "method,dim,avg_us,std_us\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

#ifndef DIM_COUNT
#define DIM_COUNT 10
#endif
#define DIM_START 1<<3
    int dims[DIM_COUNT];
    for (int i = 0; i < DIM_COUNT; i += 2) {
        dims[i] = i == 0 ? DIM_START : dims[i-2]*2;
        dims[i+1] = dims[i] + (dims[i]/2);
    }

    for (auto MAT_DIM : dims) {
        std::vector<float> dataA(MAT_DIM * MAT_DIM), dataB(MAT_DIM * MAT_DIM);
        for (auto& v : dataA) v = dis(gen);
        for (auto& v : dataB) v = dis(gen);

        MatrixCL A(MAT_DIM, MAT_DIM, context, queue, &dataA);
        MatrixCL B(MAT_DIM, MAT_DIM, context, queue, &dataB);

        MatrixCL warmup = A * B;

        double total_duration = 0.0;
        std::vector<double> durations(NUM_TESTS);
        for (int t = 0; t < NUM_TESTS; t++) {
            MatrixCL C = A * B;
            cl::Event event = C.last_event_; 

            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            double duration = (end - start) / 1000.0;
            durations[t] = duration;
            total_duration += duration;
        }

        double avg = total_duration / NUM_TESTS;
        double sigma = 0.0;
        for (int t = 0; t < NUM_TESTS; t++)
            sigma += pow(durations[t] - avg, 2.0);
        sigma = sqrt(sigma / (NUM_TESTS - 1));

        std::cout << method << "," << MAT_DIM << "," << avg << "," << sigma << "\n";
        csv << method << "," << MAT_DIM << "," << avg << "," << sigma << "\n";
    }

    csv.close();
    std::cout << "Results written to performance/" << method << ".csv\n";
    return 0;
}