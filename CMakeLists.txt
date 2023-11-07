cmake_minimum_required(VERSION 3.17)

project(vallex.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GoogleTest)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")


option(VALLEX_BUILD_SHARED_LIBRARY "vallex: build as a shared library" OFF)
option(VALLEX_STATIC "vallex: static link libraries" OFF)
option(VALLEX_NATIVE "vallex: enable -march=native flag" OFF)
option(VALLEX_LTO "vallex: enable link time optimization" OFF)

# ggml
include(ggml)
include_directories(${ggml_SOURCE_DIR}/include)
set_target_properties(ggml PROPERTIES BUILD_SHARED_LIBS_DEFAULT OFF)
set_target_properties(ggml PROPERTIES GGML_STANDALONE OFF)

if (VALLEX_BUILD_SHARED_LIBRARY)
    add_library(
            vallex
            SHARED
            transformer.h
            transformer.cpp
            embedding.h
            embedding.cpp
            vallex.h
            vallex.cpp
            common.h
            common.cpp
            activation.h
            activation.cpp
    )
    set_target_properties(vallex PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(vallex PRIVATE VALLEX_SHARED VALLEX_BUILD)
else ()
    add_library(
            vallex
            OBJECT
            transformer.h
            transformer.cpp
            embedding.h
            embedding.cpp
            vallex.h
            vallex.cpp
            common.h
            common.cpp
            activation.h
            activation.cpp
    )
endif ()

target_include_directories(vallex PUBLIC .)
target_link_libraries(vallex PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})


enable_testing()
add_subdirectory(tests)