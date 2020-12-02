cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 20)

project(ruminate VERSION 1.1.4 DESCRIPTION "an ML library that aims to be lightweight, fast, and generic. Written in and for C++20")
add_library(ruminate SHARED
    Ruminate/src/layers.cpp
    Ruminate/src/net.cpp
    Ruminate/src/HelperFiles/a_funcs.cpp
    Ruminate/src/HelperFiles/rng_helpers.cpp
    Ruminate/src/LayerModules/ann_layers.cpp
    Ruminate/src/LayerModules/generic_layers.cpp
)
set_target_properties(ruminate PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(ruminate PROPERTIES PUBLIC_HEADER
    Ruminate/include/rum_core.hpp
    Ruminate/include/layers.hpp
    Ruminate/include/net.hpp
    Ruminate/include/HelperFiles/a_funcs.hpp
    Ruminate/include/HelperFiles/rng_helpers.hpp
    Ruminate/include/LayerModules/ann_layers.hpp
    Ruminate/include/LayerModules/generic_layers.hpp
)
target_include_directories(ruminate PRIVATE .)
target_include_directories(ruminate PRIVATE Ruminate/include)
target_include_directories(ruminate PRIVATE Ruminate/src)
include(GNUInstallDirs)
install(TARGETS ruminate
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
