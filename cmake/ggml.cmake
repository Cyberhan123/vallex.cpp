
include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(GGML_GIT_TAG  ef336850d5bfe8237ebca1ec82cdfb97d78baff1)
set(GGML_GIT_URL  https://github.com/Cyberhan123/ggml.git)

FetchContent_Declare(
  ggml
  GIT_REPOSITORY    ${GGML_GIT_URL}
  GIT_TAG           ${GGML_GIT_TAG}
)

FetchContent_MakeAvailable(ggml)