cmake_minimum_required(VERSION 3.20)
project(xpcs_analysis)

set(CMAKE_CXX_STANDARD 17)

add_library(xpcs_analysis STATIC library.cpp)
add_executable(main main.cpp)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# submodules
add_subdirectory(extern/argparse)
include_directories(extern/argparse/include)

find_package(OpenMP REQUIRED COMPONENTS CXX)
#find_package(carma CONFIG REQUIRED)
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS CXX)

message(STATUS "HDF5 Parallel support:" ${HDF5_IS_PARALLEL})

target_link_libraries(
        xpcs_analysis
        PUBLIC
        OpenMP::OpenMP_CXX
        ${ARMADILLO_LIBRARIES}
        HDF5::HDF5
)

target_link_libraries(
        main
        PRIVATE
        xpcs_analysis
        OpenMP::OpenMP_CXX
        ${ARMADILLO_LIBRARIES}
        HDF5::HDF5
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
