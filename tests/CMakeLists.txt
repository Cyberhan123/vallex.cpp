cmake_minimum_required(VERSION 3.17)
project(vallex_test)

include(googletest)
include_directories(
        ${googletest_SOURCE_DIR}/googletest/include
)
function(vallex_add_test source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    target_link_libraries(${TEST_TARGET} PRIVATE GTest::gtest_main ggml vallex)
    gtest_discover_tests(${TEST_TARGET})
endfunction()

vallex_add_test(test_embedding.cpp)