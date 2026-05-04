#!/bin/bash

DIM_COUNT=16

g++ -DUSE_AVX2 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -mavx2 -mfma -march=native && ./matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=1 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=2 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=4 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=8 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=16 -DDIM_COUNT=$DIM_COUNT -o matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./matrix_perf

mpic++ -DDIM_COUNT=$DIM_COUNT -o distributed_matrix_perf src/distributed_matrix_perf.cpp src/distributed_matrix.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 && mpirun -np 4 ./distributed_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=0 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=4 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=8 -o opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./opencl_matrix_perf