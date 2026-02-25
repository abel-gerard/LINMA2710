// C++ Program to implement
// Parallel Programming
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>

// Computes the value of pi using a serial computation.

double compute_pi_serial(long num_steps)
{
    double sum = 0.0;
    for (size_t i = 1; i <= num_steps; i++) {
        double sign = (i%2) == 0 ? -1. : 1.;
        sum += sign/(2.*i-1.);
    }

    return 4.*sum;
}

double compute_pi_parallel(long num_steps, size_t num_threads=4UL)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
    for (size_t i = 1; i <= num_steps; i++) {
        double sign = (i%2) == 0 ? -1. : 1.;
        sum += sign/(2.*i-1.);
    }

    return 4.*sum;
}

// Driver function
int main()
{
    // const long num_steps = 1'000'000'000L;

    // // Compute pi using serial computation and time it.
    // auto start_time
    //     = std::chrono::high_resolution_clock::now();
    // double pi_serial = compute_pi_serial(num_steps);
    // auto end_time
    //     = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> serial_duration
    //     = end_time - start_time;

    // // Compute pi using parallel computation and time it.
    // start_time = std::chrono::high_resolution_clock::now();
    // double pi_parallel = compute_pi_parallel(num_steps);
    // end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> parallel_duration
    //     = end_time - start_time;

    // std::cout << "Serial result: " << pi_serial
    //           << std::endl;
    // std::cout << "Parallel result: " << pi_parallel
    //           << std::endl;
    // std::cout << "Serial duration: "
    //           << serial_duration.count() << " seconds"
    //           << std::endl;
    // std::cout << "Parallel duration: "
    //           << parallel_duration.count() << " seconds"
    //           << std::endl;
    // std::cout << "Speedup: "
    //           << serial_duration.count()
    //                  / parallel_duration.count()
    //           << std::endl;
    
    std::fstream fs;
    fs.open("perf.csv", std::ios::out);
    fs << "# Threads, # Terms, Serial Time, Parallel Time\n";
    
    for (size_t num_threads = 1; num_threads <= 16; num_threads *= 2) {
        for (size_t num_terms = 1000; num_terms <= 1'000'000'000UL; num_terms *= 1000) {
            auto start_time = std::chrono::high_resolution_clock::now();
            double pi_serial = compute_pi_serial(num_terms);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> serial_duration = end_time - start_time;

            start_time = std::chrono::high_resolution_clock::now();
            double pi_parallel = compute_pi_parallel(num_terms, num_threads);
            end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> parallel_duration = end_time - start_time;

            fs << num_threads << ", " << num_terms << ", " << serial_duration.count() << ", " << parallel_duration.count() << std::endl;
        }
    }

    fs.close();
    return 0;
}