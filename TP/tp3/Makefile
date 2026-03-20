all: query_device gpu_square

query_device: query_device.cpp
	g++ -std=c++17 -Iinclude -O2 -o query_device query_device.cpp -lOpenCL 

gpu_square: gpu_square.cpp
	g++ -std=c++17 -Iinclude -O2 -o gpu_square gpu_square.cpp -lOpenCL 