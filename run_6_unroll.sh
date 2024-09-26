#! /bin/bash

g++ -O3 -g -mavx2 -march=native ./6_unroll.cc && ./a.out
