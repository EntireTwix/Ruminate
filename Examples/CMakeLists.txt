cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(Ruminate_examples)

# Build binaries for the examples
add_executable(Ruminate_examples ${CMAKE_CURRENT_LIST_DIR}/addition.cpp)
add_executable(Ruminate_examples ${CMAKE_CURRENT_LIST_DIR}/batchs.cpp)
add_executable(Ruminate_examples ${CMAKE_CURRENT_LIST_DIR}/dropout.cpp)