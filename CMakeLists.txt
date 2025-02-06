cmake_minimum_required(VERSION 3.15)

include(FetchContent)

project(noarr_mpi
        VERSION 1.0
        DESCRIPTION "noarr-mpi"
        LANGUAGES CXX C)

FetchContent_Declare(Noarr
                    GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
                    GIT_TAG        experimental)
FetchContent_MakeAvailable(Noarr)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)

add_executable(mpi-transform src/mpi_transform.cpp)
target_link_libraries(mpi-transform MPI::MPI_CXX)

add_executable(gemm examples/gemm/gemm.cpp)
target_compile_definitions(gemm PRIVATE EXTRALARGE_DATASET DATA_TYPE_IS_FLOAT)

add_executable(gemm-mpi examples/gemm/gemm-mpi.cpp)
target_compile_definitions(gemm-mpi PRIVATE EXTRALARGE_DATASET DATA_TYPE_IS_FLOAT)
target_link_libraries(gemm-mpi MPI::MPI_CXX)

add_executable(gemm-mpi-tilea-transpose examples/gemm/gemm-mpi.cpp)
target_compile_definitions(gemm-mpi-tilea-transpose PRIVATE EXTRALARGE_DATASET DATA_TYPE_IS_FLOAT A_TILE_K_MAJOR)
target_link_libraries(gemm-mpi-tilea-transpose MPI::MPI_CXX)

# ask the compiler to print maximum warnings
if(MSVC)
  target_compile_options(mpi-transform PRIVATE /W4)
  target_compile_options(gemm PRIVATE /W4)
  target_compile_options(gemm-mpi PRIVATE /W4)
  target_compile_options(gemm-mpi-tilea-transpose PRIVATE /W4)
else()
  target_compile_options(mpi-transform PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
  target_compile_options(gemm PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
  target_compile_options(gemm-mpi PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
  target_compile_options(gemm-mpi-tilea-transpose PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
endif()
