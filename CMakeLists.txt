cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 20)

#project(ruminate)
#add_executable(${PROJECT_NAME} ${SRC} main.cpp)

include_directories(PUBLIC include)
include_directories(PUBLIC third_party) #for OptimizedHeaders & pcg32