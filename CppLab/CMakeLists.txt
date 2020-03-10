cmake_minimum_required(VERSION 2.8.3)
project(FlowLab)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE "Debug")

# OpenCV
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
message(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(${EIGEN3_INCLUDE_DIR})

# g2o
find_package(g2o REQUIRED)
message("G2O lib is found:"${g2o_INCLUDE_DIR})

# openmp
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# thread
find_package(Threads)
#gtest
find_package(GTest REQUIRED)

add_library(util
    util/show_flow.cpp
    util/read_write.cpp
    util/evaluation.cpp
    util/line_tracking.cpp
)
target_link_libraries(util
    ${OpenCV_LIBS}
)

# EXECUTE
add_executable(Franeback_Lab
    src/Farneback.cpp
)
target_link_libraries(Franeback_Lab
    util
    ${OpenCV_LIBS}
)

add_executable(DeepFlow_Lab
    src/DeepFlow.cpp
)
target_link_libraries(DeepFlow_Lab
    util
    ${OpenCV_LIBS}
)

# TESTs
add_executable(read_write_TEST
    util/test/read_write_TEST.cpp
)
target_link_libraries(read_write_TEST
    util
    ${OpenCV_LIBS}
)

add_executable(evaluation_TEST
    util/test/evaluation_TEST.cpp
)
target_link_libraries(evaluation_TEST
    util
    ${OpenCV_LIBS}
)

add_executable(line_tracking_TEST
    util/test/line_tracking_TEST.cpp
)
target_link_libraries(line_tracking_TEST
    util
    ${OpenCV_LIBS}
)
