#include "matrix.hpp"
#include "matrix_opencl.hpp"
#include <iostream>
#include <cassert>
#include <random>
#include <vector>


cl::Context context;
cl::CommandQueue queue;

void setupOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty());

    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    assert(!devices.empty());

    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    MatrixCL::initializeKernels(context, {device});

    std::cout << "setupOpenCL passed." << std::endl;
}


int main() {
    setupOpenCL();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    const int M = 36, N = 23, O = 15;

    std::vector<float> dataA(M * N), dataB(N * O);
    for (auto& v : dataA) v = dis(gen);
    for (auto& v : dataB) v = dis(gen);

    Matrix matA(M, N), matB(N, O);
    MatrixCL A(M, N, context, queue, &dataA), B(N, O, context, queue, &dataB);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matA.set(i, j, dataA[i*N+j]);
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < O; j++) {
            matB.set(i, j, dataB[i*O+j]);
        }
    }

    Matrix matResult = matA * matB;
    std::vector<float> result = (A * B).copyToHost();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < O; j++) {
            assert(fabsf(matResult.get(i, j) - result[i*O+j]) < 1e-4);
        }
    }

    printf("It's all good :-)\n");

    return 0;
}