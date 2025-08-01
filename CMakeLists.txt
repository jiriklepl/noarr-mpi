cmake_minimum_required(VERSION 3.20)

project(noarr_mpi
        VERSION 1.0
        DESCRIPTION "noarr-mpi"
        LANGUAGES CXX C)

include(FetchContent)
include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NOARR_MPI_EXAMPLES Default CACHE STRING "Examples to build")
set_property(CACHE NOARR_MPI_EXAMPLES PROPERTY STRINGS Default AllDatasets All Debug Debug2 Debug3 Debug4)

if (NOT DEFINED NOARR_MPI_EXAMPLES)
  set(NOARR_MPI_EXAMPLES "Default")
endif()

# Fetch Noarr Structures
FetchContent_Declare(Noarr
                    GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
                    GIT_TAG        888bd796955bae9cd2f156b6a5aff250751a5b5e)
FetchContent_MakeAvailable(Noarr)

find_package(MPI REQUIRED)

if (NOARR_MPI_EXAMPLES STREQUAL "Default")
  message(STATUS "Building the default examples")
  set(DATASET_LIST "MINI;MEDIUM;EXTRALARGE")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "AllDatasets")
  message(STATUS "Building the examples for all datasets")
  set(DATASET_LIST "MINI;SMALL;MEDIUM;LARGE;EXTRALARGE")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "All")
  message(STATUS "Building all examples")
  set(DATASET_LIST "MINI;SMALL;MEDIUM;LARGE;EXTRALARGE")
  set(DATATYPE_LIST "FLOAT;DOUBLE;INT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "Debug")
  message(STATUS "Building the Debug examples")
  set(DATASET_LIST "SMALL")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "Debug2")
  message(STATUS "Building the Debug2 examples")
  set(DATASET_LIST "SMALL")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "Debug3")
  message(STATUS "Building the Debug3 examples")
  set(DATASET_LIST "MEDIUM")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
elseif (NOARR_MPI_EXAMPLES STREQUAL "Debug4")
  message(STATUS "Building the Debug4 examples")
  set(DATASET_LIST "MINI;MEDIUM")
  set(DATATYPE_LIST "FLOAT")
  set(C_TILE_OPTIONS "C_TILE_I_MAJOR;C_TILE_J_MAJOR")
  set(A_TILE_OPTIONS "A_TILE_I_MAJOR;A_TILE_K_MAJOR")
  set(B_TILE_OPTIONS "B_TILE_K_MAJOR;B_TILE_J_MAJOR")

  set(C_SCATTER_OPTIONS "C_SCATTER_J_MAJOR;C_SCATTER_I_MAJOR")
  set(A_SCATTER_OPTIONS "A_SCATTER_I_MAJOR;A_SCATTER_K_MAJOR")
  set(B_SCATTER_OPTIONS "B_SCATTER_K_MAJOR;B_SCATTER_J_MAJOR")
  # set(C_GATHER_OPTIONS "C_GATHER_J_MAJOR;C_GATHER_I_MAJOR")
else ()
  message(FATAL_ERROR "Unknown NOARR_MPI_EXAMPLES value: ${NOARR_MPI_EXAMPLES}")
endif ()

function(define_executable name framework)
  foreach(dataset ${DATASET_LIST})
    foreach(datatype ${DATATYPE_LIST})
      foreach(c_tile ${C_TILE_OPTIONS})
        foreach(a_tile ${A_TILE_OPTIONS})
          foreach(b_tile ${B_TILE_OPTIONS})
            set(executable_name ${name}-${framework}-${dataset}-${datatype}-${c_tile}-${a_tile}-${b_tile})
            add_executable(${executable_name} ${name}.cpp)
            target_compile_definitions(${executable_name} PRIVATE
              ${dataset}_DATASET
              DATA_TYPE_IS_${datatype}
              ${c_tile}
              ${a_tile}
              ${b_tile})
            target_include_directories(${executable_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
            target_include_directories(${executable_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
            target_link_libraries(${executable_name} PRIVATE MPI::MPI_CXX)
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endfunction()

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)

add_subdirectory(examples/boost)
add_subdirectory(examples/boostP2P)
add_subdirectory(examples/kokkosComm)
add_subdirectory(examples/mpi)
add_subdirectory(examples/noarr)

foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    set(test_file_name examples/test/gemm-${dataset}-${datatype}.data)
    configure_file(${test_file_name} ${CMAKE_BINARY_DIR}/gemm-${dataset}-${datatype}.data COPYONLY)
  endforeach()
endforeach()

#tests
add_executable(auto-permutate tests/auto-permutate.cpp)
target_link_libraries(auto-permutate PRIVATE MPI::MPI_CXX)
add_test(NAME auto-permutate COMMAND auto-permutate)

add_executable(reverse tests/reverse.cpp)
target_link_libraries(reverse PRIVATE MPI::MPI_CXX)
add_test(NAME reverse COMMAND reverse)

add_executable(slice tests/slice.cpp)
target_link_libraries(slice PRIVATE MPI::MPI_CXX)
add_test(NAME slice COMMAND slice)

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
