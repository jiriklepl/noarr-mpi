cmake_minimum_required(VERSION 3.20)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)

find_package(KokkosComm REQUIRED)

add_executable(gemm-kokkosComm gemm.cpp)
target_compile_definitions(gemm-kokkosComm PRIVATE ${DATASET}_DATASET DATA_TYPE_IS_${DATATYPE})
target_include_directories(gemm-kokkosComm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(gemm-kokkosComm PRIVATE KokkosComm::KokkosComm)
