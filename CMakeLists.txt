cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(Ruminate)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Library directory
set(VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Ruminate)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Examples)
