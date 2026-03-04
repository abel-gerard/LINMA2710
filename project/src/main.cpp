#include <chrono>
#include <iostream>
#include <random>

#include "matrix.hpp"

#define MAT_DIM 1<<10
#define NUM_TESTS 10.


int main() 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);
    
    Matrix A(MAT_DIM, MAT_DIM), B(MAT_DIM, MAT_DIM);
    std::chrono::duration<double> total_duration(0);
    for (int t = 0; t < NUM_TESTS; t++) {
        for (int i = 0; i < MAT_DIM; i++) {
            for (int j = 0; j < MAT_DIM; j++) {
                A.set(i, j, dis(gen));
                B.set(i, j, dis(gen));
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        Matrix C = A * B;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        total_duration += duration;
    }
    
    std::cout << std::endl;
    std::cout << "Multiplication in " << total_duration.count()/NUM_TESTS << "µs" << std::endl;
}