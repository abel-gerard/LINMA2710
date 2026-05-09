#!/bin/bash

if [ -z "$DIM_COUNT" ]; then
    echo "Warning: DIM_COUNT is not set. Defaulting to 10."
    DIM_COUNT=10
fi

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

g++ -DUSE_AVX2 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=1 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=2 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=4 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=8 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf
g++ -DUSE_AVX2 -DUSE_OMP -DOMP_T=16 -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/matrix_perf src/matrix_perf.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 -fopenmp -mavx2 -mfma -march=native && ./$BUILD_DIR/matrix_perf

mpic++ -DDIM_COUNT=$DIM_COUNT -o $BUILD_DIR/distributed_matrix_perf src/distributed_matrix_perf.cpp src/distributed_matrix.cpp src/matrix.cpp -Iinclude -std=c++17 -Wall -Wextra -O2 && mpirun -np 4 ./$BUILD_DIR/distributed_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=0 -o $BUILD_DIR/opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./$BUILD_DIR/opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -o $BUILD_DIR/opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./$BUILD_DIR/opencl_matrix_perf

g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=4 -o $BUILD_DIR/opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./$BUILD_DIR/opencl_matrix_perf
g++ -DDIM_COUNT=$DIM_COUNT -DCL_MUL_METHOD=1 -DTILE_SIZE=8 -o $BUILD_DIR/opencl_matrix_perf src/opencl_matrix_perf.cpp src/matrix_opencl.cpp -std=c++17 -Wall -Wextra -O2 -Iinclude -lOpenCL && ./$BUILD_DIR/opencl_matrix_perf
