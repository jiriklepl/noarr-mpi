cmake_minimum_required(VERSION 3.15)

include(FetchContent)

project(noarr_mpi
        VERSION 1.0
        DESCRIPTION "noarr-mpi (WIP)"
        LANGUAGES CXX)

FetchContent_Declare(Noarr
                    GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
                    GIT_TAG        experimental)
FetchContent_MakeAvailable(Noarr)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)

add_executable(mpi-transform src/mpi_transform.cpp)
add_executable(matmul src/matmul.cpp)

# ask the compiler to print maximum warnings
if(MSVC)
  target_compile_options(mpi-transform PRIVATE /W4)
  target_compile_options(matmul PRIVATE /W4)
else()
  target_compile_options(mpi-transform PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
  target_compile_options(matmul PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
endif()

target_link_libraries(mpi-transform MPI::MPI_CXX)
target_link_libraries(matmul MPI::MPI_CXX)
