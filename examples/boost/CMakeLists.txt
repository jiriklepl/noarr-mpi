cmake_minimum_required(VERSION 3.20)

project(noarr_mpi
        DESCRIPTION "boost example"
        LANGUAGES CXX C)

include(FetchContent)

FetchContent_Declare(mdspan
                     GIT_REPOSITORY https://github.com/kokkos/mdspan.git
                     GIT_TAG        mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

find_package(Boost REQUIRED  COMPONENTS mpi system serialization)
find_package(MPI REQUIRED)

add_executable(gemm-boost gemm.cpp)
target_compile_definitions(gemm-boost PRIVATE ${DATASET}_DATASET DATA_TYPE_IS_${DATATYPE})
target_include_directories(gemm-boost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gemm-boost PRIVATE ${mdspan_SOURCE_DIR}/include)

target_link_libraries(gemm-boost PRIVATE MPI::MPI_CXX)
target_link_libraries(gemm-boost PRIVATE boost_serialization boost_mpi boost_system)
