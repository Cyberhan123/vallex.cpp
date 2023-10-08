project(vallex.cpp)
cmake_minimum_required(VERSION 3.11)

add_definitions(-std=c++11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(ggml)

include_directories(${ggml_SOURCE_DIR}/include)

add_library(vallex
            SHARED
            vallex.cpp
            vallex.h
)

target_include_directories(vallex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vallex ggml ${CMAKE_THREAD_LIBS_INIT})