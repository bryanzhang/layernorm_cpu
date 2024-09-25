#! /bin/bash

g++ -O3 -g -mavx2 -march=native ./3_fma.cc && ./a.out
