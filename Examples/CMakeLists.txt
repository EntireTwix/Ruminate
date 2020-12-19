cmake_minimum_required(VERSION 3.13)
project(RuminateExamples)

add_executable(addition "addition.cpp")
add_executable(batchs "batchs.cpp")
add_executable(dropout "dropout.cpp")

target_link_libraries(addition Ruminate)
target_link_libraries(batchs Ruminate)
target_link_libraries(dropout Ruminate)
