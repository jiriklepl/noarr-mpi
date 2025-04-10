cmake_minimum_required(VERSION 3.20)

set(BUILD_TESTING OFF CACHE BOOL "Disable testing")

# Fetch MPL
FetchContent_Declare(mpl
                     GIT_REPOSITORY https://github.com/rabauke/mpl.git
                     GIT_TAG        master)
FetchContent_MakeAvailable(mpl)

define_executable(gemm mpl)
foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    foreach(c_tile ${C_TILE_OPTIONS})
      foreach(a_tile ${A_TILE_OPTIONS})
        foreach(b_tile ${B_TILE_OPTIONS})
          set(executable_name gemm-mpl-${dataset}-${datatype}-${c_tile}-${a_tile}-${b_tile})
          target_link_libraries(${executable_name} PRIVATE mpl::mpl)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
