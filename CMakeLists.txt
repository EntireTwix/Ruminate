cmake_minimum_required(VERSION 3.10)
project(ruminate)
set (CMAKE_CXX_STANDARD 20)

file(GLOB SRC "src/header_defs/*.cpp src/modules/*.cpp")
# option(ANN_LAYERS "Build the ANN layers" ON)
# if(ANN_LAYERS)
#     string(APPEND SRC "src/modules/ann_layers.hpp")
# endif(ANN_LAYERS)

add_executable(${PROJECT_NAME} ${SRC} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC third_party) #for OptimizedHeaders & pcg32g