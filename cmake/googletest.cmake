include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if (${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif ()

if (${CMAKE_VERSION} VERSION_GREATER 3.23)
    cmake_policy(SET CMP0135 OLD)
endif ()

set(GTEST_URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)

FetchContent_Declare(
        googletest
        URL ${GTEST_URL}
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)