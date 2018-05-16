#!/usr/bin/env bash

set -x
set -e

#GCC_OPT_FLAGS="-O3 -flto=8 -fno-fat-lto-objects -march=native"
GCC_OPT_FLAGS="-O0 -march=native"
GCC_WARNING_FLAGS="-Wall -Wextra -pedantic -Werror"
GCC_FLAGS="$GCC_OPT_FLAGS $GCC_WRN_FLAGS"
EXE_NAME="rksearch-O3-lto"

mkdir -p obj
mkdir -p bin

g++-7 -std=c++17 $GCC_FLAGS -c objective_function.cpp -o obj/objective_function.o
g++-7 -std=c++17 $GCC_FLAGS -c bfgs_subroutines.cpp -o obj/bfgs_subroutines.o
g++-7 -std=c++17 $GCC_OPT_FLAGS rksearch_main.cpp obj/*.o -o bin/$EXE_NAME -lmpfr -lgmp
rm obj/*.o
