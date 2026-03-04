#!/bin/bash

clang++ -o matrix src/main.cpp src/matrix.cpp -Iinclude -O3 -ffast-math -march=native -mtune=native -mavx2 && sudo perf stat -e cycles,instructions,cache-references,cache-misses ./matrix
