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

define_executable(gemm boost)
foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    foreach(c_tile ${C_TILE_OPTIONS})
      foreach(a_tile ${A_TILE_OPTIONS})
        foreach(b_tile ${B_TILE_OPTIONS})
          set(executable_name gemm-boost-${dataset}-${datatype}-${c_tile}-${a_tile}-${b_tile})
          target_include_directories(${executable_name} PRIVATE ${mdspan_SOURCE_DIR}/include)
          target_link_libraries(${executable_name} PRIVATE boost_serialization boost_mpi boost_system)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
