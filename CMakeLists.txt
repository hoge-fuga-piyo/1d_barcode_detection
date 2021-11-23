###############################
# Compiler setting
###############################

cmake_minimum_required(VERSION 3.0)

###############################
# External libraries
###############################
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

###############################
# For unit test
###############################
# Google Test
#enable_testing()
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})

###############################
# Sources
###############################
set(PROJ_SHARED_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB PROJ_SHARED_SOURCE ${PROJ_SHARED_DIR}/*.cpp)
file(GLOB PROJ_SHARED_INCLUDE ${PROJ_SHARED_DIR}/*.h ${PROJ_SHARED_DIR}/*.hpp)

###############################
# Project
###############################
set(PROJ_SUB_DIR ${CMAKE_SOURCE_DIR}/project)
add_subdirectory(${PROJ_SUB_DIR})
