cmake_minimum_required(VERSION 3.30)
project(Assignment2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)

# CPU
add_executable(GaussJordanC
        gaussjordanseq.cpp
    )

add_executable(OddEvenSortC
    oddevensort.cpp
    )

# CUDA

set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G --expt-relaxed-constexpr")

set(ODDEVEN_HEADERS oddevensort.cuh)
add_executable(OddEvenSortCuda
    oddevensort.cu
    oddevensortmain.cu
    ${ODDEVEN_HEADERS}
    )

set(GAUSSJORDAN_HEADERS gaussjordan.cuh)
add_executable(GaussJordanCuda
        gaussjordan.cu
        gaussjordanmain.cu
    )