cmake_minimum_required(VERSION 3.10)
project(rktkm)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(rktkm
        bfgs_subroutines.cpp
        bfgs_subroutines.hpp
        nonlinear_optimizers.hpp
        objective_function.cpp
        objective_function.hpp
        OrderConditionHelpers.cpp
        OrderConditionHelpers.hpp
        rksearch_main.cpp MPFRVector.hpp MPFRMatrix.hpp FilenameHelpers.hpp QuadraticLineSearcher.hpp)
