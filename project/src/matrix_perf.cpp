#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

#include "matrix.hpp"

#define NUM_TESTS 10

#ifdef USE_OMP
#if OMP_T == 1
    const std::string method = "omp_t1";
#elif OMP_T == 2
    const std::string method = "omp_t2";
#elif OMP_T == 4
    const std::string method = "omp_t4";
#elif OMP_T == 8
    const std::string method = "omp_t8";
#elif OMP_T == 16
    const std::string method = "omp_t16";
#else
#define OMP_T 1
    const std::string method = "omp_t1";
#endif

#else
    const std::string method = "matrix";
#endif

int main() 
{
    std::filesystem::create_directories("performance");
    std::ofstream csv("performance/" + method + ".csv", std::ios::trunc);
    csv << "method,dim,avg_us,std_us\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);
    
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
        Matrix A(MAT_DIM, MAT_DIM), B(MAT_DIM, MAT_DIM);
        for (int i = 0; i < MAT_DIM; i++) {
            for (int j = 0; j < MAT_DIM; j++) {
                A.set(i, j, dis(gen));
                B.set(i, j, dis(gen));
            }
        }

        Matrix warmup = A * B;

        double total_duration = 0.0f;
        std::vector<double> durations(NUM_TESTS);
        for (int t = 0; t < NUM_TESTS; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            Matrix C = A * B;
            auto end = std::chrono::high_resolution_clock::now();

            auto ch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double duration = ch_duration.count();
            total_duration += duration;
            durations[t] = duration;
        }

        double avg = total_duration / NUM_TESTS;
        double sigma = 0.0f;
        for (int t = 0; t < NUM_TESTS; t++) {
            sigma += pow(durations[t] - avg, 2.0f);
        }
        sigma = sqrt(sigma / (NUM_TESTS - 1));

        std::cout << method << "," << MAT_DIM << "," << avg << "," << sigma << "\n";
        csv << method << "," << MAT_DIM << "," << avg << "," << sigma << "\n";
    }

    csv.close();
    std::cout << "Results written to performance/" << method << ".csv\n";
    return 0;
}