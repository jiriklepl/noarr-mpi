cmake_minimum_required(VERSION 3.20)

# Fetch Noarr Structures
FetchContent_Declare(Noarr
                    GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
                    GIT_TAG        main)
FetchContent_MakeAvailable(Noarr)

find_package(MPI REQUIRED)

foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    set(executable_name gemm-nompi-${dataset}-${datatype})
    add_executable(${executable_name} gemm-nompi.cpp)
    target_compile_definitions(${executable_name} PRIVATE ${dataset}_DATASET DATA_TYPE_IS_${datatype})
    target_include_directories(${executable_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_include_directories(${executable_name} PRIVATE ${Noarr_SOURCE_DIR}/include)
  endforeach()
endforeach()

define_executable(gemm noarr)
foreach(dataset ${DATASET_LIST})
  foreach(datatype ${DATATYPE_LIST})
    foreach(c_tile ${C_TILE_OPTIONS})
      foreach(a_tile ${A_TILE_OPTIONS})
        foreach(b_tile ${B_TILE_OPTIONS})
          set(executable_name gemm-noarr-${dataset}-${datatype}-${c_tile}-${a_tile}-${b_tile})
          target_include_directories(${executable_name} PRIVATE ${Noarr_SOURCE_DIR}/include)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
