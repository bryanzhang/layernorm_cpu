#! /bin/bash

g++ -O3 -g -mavx512f -march=native ./2_simd16.cc && ./a.out
