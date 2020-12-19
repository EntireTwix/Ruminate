cmake_minimum_required(VERSION 3.13)

project(libRuminate)

set(SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)

# Get sources
file(GLOB_RECURSE SRC ${SOURCE_DIR}/*.cpp)

add_library(Ruminate ${SRC})
target_include_directories(Ruminate PUBLIC ${INCLUDE_DIR})

# Libraries
target_include_directories(Ruminate PUBLIC ${VENDOR_DIR}/pcg32)
target_include_directories(Ruminate PUBLIC ${VENDOR_DIR}/OptimizedHeaders)
