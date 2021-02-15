cmake_minimum_required(VERSION 3.10)
project(ruminate)
set (CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} ${SRC} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC third_party) #for OptimizedHeaders & pcg32