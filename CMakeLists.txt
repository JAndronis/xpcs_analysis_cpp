cmake_minimum_required(VERSION 3.20)
project(xpcs_analysis)

set(CMAKE_CXX_STANDARD 17)

# submodules
add_subdirectory(extern/argparse)
add_subdirectory(extern/eigen)
add_subdirectory(extern/pybind11)
include_directories(extern/pybind11/include)

# dependencies
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(pybind11 CONFIG)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-math-errno")

pybind11_add_module(xpcs_analysis_py MODULE ttc.cpp ttc.h g2.cpp g2.h bindings.cpp)
target_link_libraries(xpcs_analysis_py PRIVATE pybind11::module Eigen3::Eigen OpenMP::OpenMP_CXX)

install(TARGETS xpcs_analysis_py DESTINATION .)
