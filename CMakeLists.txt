cmake_minimum_required(VERSION 3.4)
project(snnow)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set (THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/thirdparty)

include(${THIRDPARTY_DIR}/dmlc-mshadow/cmake/mshadow.cmake)
include(${THIRDPARTY_DIR}/dmlc-mshadow/cmake/Utils.cmake)
include(${THIRDPARTY_DIR}/dmlc-mshadow/cmake/Cuda.cmake)

include_directories(${SOURCE_DIR}/include)
include_directories(${THIRDPARTY_DIR}/mshadow/mshadow)

MESSAGE(STATUS "This is include directory" ${CMAKE_INCLUDE_PATH})
add_subdirectory(src bin)

