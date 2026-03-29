#!/bin/bash

clang++ -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && ./matrix_perf #&& sudo perf stat -e cycles,instructions,cache-references,cache-misses ./matrix_perf
mpic++ -o distributed_matrix_perf src/distributed_matrix_perf.cpp src/distributed_matrix.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && mpirun -np 4 ./distributed_matrix_perf
