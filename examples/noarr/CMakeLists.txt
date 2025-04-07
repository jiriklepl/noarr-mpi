cmake_minimum_required(VERSION 3.15)

# Fetch Noarr Structures
FetchContent_Declare(Noarr
                    GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
                    GIT_TAG        main)
FetchContent_MakeAvailable(Noarr)

find_package(MPI REQUIRED)

add_executable(gemm-noarr-nompi gemm.cpp)
target_compile_definitions(gemm-noarr-nompi PRIVATE ${DATASET}_DATASET DATA_TYPE_IS_${DATATYPE})
target_include_directories(gemm-noarr-nompi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gemm-noarr-nompi PRIVATE ${Noarr_SOURCE_DIR}/include)

add_executable(gemm-noarr gemm-mpi.cpp)
target_compile_definitions(gemm-noarr PRIVATE ${DATASET}_DATASET DATA_TYPE_IS_${DATATYPE})
target_include_directories(gemm-noarr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gemm-noarr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(gemm-noarr PRIVATE ${Noarr_SOURCE_DIR}/include)
target_link_libraries(gemm-noarr PRIVATE MPI::MPI_CXX)
