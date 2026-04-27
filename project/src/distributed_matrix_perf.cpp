#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

#include "distributed_matrix.hpp"

#define NUM_TESTS 10

int main(int argc, char **argv) 
{
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
        MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    std::ofstream csv;
    if (rank == 0) {
        std::filesystem::create_directories("performance");
        csv.open("performance/distributed_matrix.csv", std::ios::trunc);
        csv << "method,dim,avg_us,std_us\n";
    }

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

        DistributedMatrix distB(B, numProcs);

        Matrix warmup = multiply(A, distB).gather();

        double total_duration = 0.0f;
        std::vector<double> durations(NUM_TESTS);
        for (int t = 0; t < NUM_TESTS; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            Matrix C = multiply(A, distB).gather();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto ch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double duration = ch_duration.count();
            total_duration += duration;
            durations[t] = duration;
        }

        if (rank == 0) {
            double avg = total_duration / NUM_TESTS;
            double sigma = 0.0f;
            for (int t = 0; t < NUM_TESTS; t++) {
                sigma += pow(durations[t] - avg, 2.0f);
            }
            sigma = sqrt(sigma / (NUM_TESTS - 1));

            std::cout << "distributed," << MAT_DIM << "," << avg << "," << sigma << "\n";
            csv << "distributed," << MAT_DIM << "," << avg << "," << sigma << "\n";
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Results written to performance/distributed_matrix.csv\n";
    }

    MPI_Finalize();
    return 0;
}