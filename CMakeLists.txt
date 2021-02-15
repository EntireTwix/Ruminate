cmake_minimum_required(VERSION 3.10)
project(Ruminate)
set (CMAKE_CXX_STANDARD 20)

file(GLOB SRC 
    "src/header_defs/*.cpp"
    )
add_executable(${PROJECT_NAME} ${SRC} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC src)