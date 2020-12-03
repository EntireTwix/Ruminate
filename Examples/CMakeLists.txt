cmake_minimum_required(VERSION 3.13)
project(RuminateExamples)

link_libraries(Ruminate)

add_executable(addition "addition/main.cpp")
add_executable(batchs "batchs/main.cpp")
add_executable(dropout "dropout/main.cpp")
