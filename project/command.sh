#!/bin/bash

DIM_COUNT=10

g++ -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && ./matrix_perf
g++ -DUSE_OMP -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && ./matrix_perf

mpic++ -DDIM_COUNT=$DIM_COUNT -o distributed_matrix_perf src/distributed_matrix_perf.cpp src/distributed_matrix.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && mpirun -np 4 ./distributed_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=0 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=4 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=8 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf