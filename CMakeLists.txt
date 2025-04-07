cmake_minimum_required(VERSION 3.15)

project(noarr_mpi
        VERSION 1.0
        DESCRIPTION "noarr-mpi"
        LANGUAGES CXX C)

include(FetchContent)
include(CTest)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DATASET "SMALL" CACHE STRING "Dataset size")
set_property(CACHE DATASET PROPERTY STRINGS SMALL MEDIUM LARGE EXTRALARGE)

set(DATATYPE "FLOAT" CACHE STRING "Data type")
set_property(CACHE DATATYPE PROPERTY STRINGS FLOAT DOUBLE INT)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)

add_subdirectory(examples/mpl)
add_subdirectory(examples/noarr)
add_subdirectory(examples/kokkos-comm)

#tests
add_executable(scatter-type-safe-1 EXCLUDE_FROM_ALL tests/scatter-type-safe-1.cpp)
target_link_libraries(scatter-type-safe-1 PRIVATE MPI::MPI_CXX)
add_test(NAME scatter-type-safe-1 COMMAND ${CMAKE_COMMAND} --build . --target scatter-type-safe-1 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(scatter-type-safe-1 PROPERTIES WILL_FAIL TRUE)

add_executable(scatter-type-safe-2 EXCLUDE_FROM_ALL tests/scatter-type-safe-2.cpp)
target_link_libraries(scatter-type-safe-2 PRIVATE MPI::MPI_CXX)
add_test(NAME scatter-type-safe-2 COMMAND ${CMAKE_COMMAND} --build . --target scatter-type-safe-2 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(scatter-type-safe-2 PROPERTIES WILL_FAIL TRUE)

add_executable(scatter-type-safe-3 EXCLUDE_FROM_ALL tests/scatter-type-safe-3.cpp)
target_link_libraries(scatter-type-safe-3 PRIVATE MPI::MPI_CXX)
add_test(NAME scatter-type-safe-3 COMMAND ${CMAKE_COMMAND} --build . --target scatter-type-safe-3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(scatter-type-safe-3 PROPERTIES WILL_FAIL TRUE)

add_executable(gather-type-safe-1 EXCLUDE_FROM_ALL tests/gather-type-safe-1.cpp)
target_link_libraries(gather-type-safe-1 PRIVATE MPI::MPI_CXX)
add_test(NAME gather-type-safe-1 COMMAND ${CMAKE_COMMAND} --build . --target gather-type-safe-1 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(gather-type-safe-1 PROPERTIES WILL_FAIL TRUE)

add_executable(gather-type-safe-2 EXCLUDE_FROM_ALL tests/gather-type-safe-2.cpp)
target_link_libraries(gather-type-safe-2 PRIVATE MPI::MPI_CXX)
add_test(NAME gather-type-safe-2 COMMAND ${CMAKE_COMMAND} --build . --target gather-type-safe-2 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(gather-type-safe-2 PROPERTIES WILL_FAIL TRUE)

add_executable(gather-type-safe-3 EXCLUDE_FROM_ALL tests/gather-type-safe-3.cpp)
target_link_libraries(gather-type-safe-3 PRIVATE MPI::MPI_CXX)
add_test(NAME gather-type-safe-3 COMMAND ${CMAKE_COMMAND} --build . --target gather-type-safe-3 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(gather-type-safe-3 PROPERTIES WILL_FAIL TRUE)

target_compile_options(scatter-type-safe-1 PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(scatter-type-safe-2 PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(scatter-type-safe-3 PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(gather-type-safe-1 PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(gather-type-safe-2 PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(gather-type-safe-3 PRIVATE -Wall -Wextra -Wpedantic)
