cmake_minimum_required(VERSION 3.10)
project(rktkm)

set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM "C:\\Users\\Zhang\\Documents\\GitHub\\dznl")

add_executable(rktkm
        bfgs_subroutines.hpp
        nonlinear_optimizers.hpp
        objective_function.hpp
        OrderConditionHelpers.hpp
        rksearch_main.cpp FilenameHelpers.hpp)

target_link_libraries(rktkm mpfr gmp)
