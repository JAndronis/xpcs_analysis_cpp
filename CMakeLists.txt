cmake_minimum_required(VERSION 3.20)
project(xpcs_analysis)

set(CMAKE_CXX_STANDARD 17)

add_library(xpcs_analysis STATIC library.cpp)
add_executable(main main.cpp)

# submodules
add_subdirectory(extern/argparse)
add_subdirectory(extern/HighFive)
add_subdirectory(extern/eigen)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS CXX)

target_link_libraries(
        xpcs_analysis
        PRIVATE
        OpenMP::OpenMP_CXX
        HighFive
        eigen
        argparse::argparse
)

target_link_libraries(
        main
        PRIVATE
        xpcs_analysis
        OpenMP::OpenMP_CXX
        HighFive
        eigen
        argparse::argparse
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
