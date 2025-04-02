cmake_minimum_required(VERSION 3.15)

# find Boost.MPI and Boost.Serialization
find_package(Boost REQUIRED COMPONENTS mpi system serialization)
find_package(MPI REQUIRED)


add_executable(gemm-boost gemm.cpp)

target_compile_definitions(gemm-boost PRIVATE SMALL_DATASET DATA_TYPE_IS_FLOAT)
target_include_directories(gemm-boost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(gemm-boost PRIVATE boost_mpi boost_serialization)
target_link_libraries(gemm-boost PRIVATE MPI::MPI_CXX)
