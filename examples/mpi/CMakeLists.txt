cmake_minimum_required(VERSION 3.20)

include(FetchContent)

FetchContent_Declare(mdspan
                     GIT_REPOSITORY https://github.com/kokkos/mdspan.git
                     GIT_TAG        mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

find_package(MPI REQUIRED)

define_executable(gemm mpi)
foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    foreach(c_tile ${C_TILE_OPTIONS})
      foreach(a_tile ${A_TILE_OPTIONS})
        foreach(b_tile ${B_TILE_OPTIONS})
          set(executable_name gemm-mpi-${dataset}-${datatype}-${c_tile}-${a_tile}-${b_tile})
          target_include_directories(${executable_name} PRIVATE
            ${mdspan_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIRS})
          target_link_directories(${executable_name} PRIVATE
            ${Boost_LIBRARY_DIRS})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
