cmake_minimum_required(VERSION 3.16)
#set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

project(Ruminate)

# Library directory
set(VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Ruminate)
add_subdirectory("Examples")
