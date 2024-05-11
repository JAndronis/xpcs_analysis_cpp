cmake_minimum_required(VERSION 3.28)
project(xpcs_analysis)

set(CMAKE_CXX_STANDARD 20)

add_library(xpcs_analysis STATIC library.cpp
        TwoTimeCorrelator.cpp
        TwoTimeCorrelator.h)
add_executable(main main.cpp)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(carma CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

target_link_libraries(
        xpcs_analysis
        PUBLIC
        OpenMP::OpenMP_CXX
        ${ARMADILLO_LIBRARIES}
)

target_link_libraries(
        main
        PRIVATE
        xpcs_analysis
        OpenMP::OpenMP_CXX
        ${ARMADILLO_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
