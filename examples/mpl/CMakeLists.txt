cmake_minimum_required(VERSION 3.15)

set(BUILD_TESTING OFF CACHE BOOL "Disable testing")

# Fetch MPL
FetchContent_Declare(mpl
                     GIT_REPOSITORY https://github.com/rabauke/mpl.git
                     GIT_TAG        master)
FetchContent_MakeAvailable(mpl)

add_executable(gemm-mpl gemm.cpp)

target_compile_definitions(gemm-mpl PRIVATE ${DATASET}_DATASET DATA_TYPE_IS_${DATATYPE})
target_include_directories(gemm-mpl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(gemm-mpl PRIVATE mpl::mpl)
