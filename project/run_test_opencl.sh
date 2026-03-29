g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -o test_opencl tests/test_opencl.cpp src/matrix_opencl.cpp -lOpenCL
MESA_LOADER_DRIVER_OVERRIDE=radeonsi RUSTICL_ENABLE=radeonsi ./test_opencl