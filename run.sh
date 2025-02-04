#!/bin/bash

BUILD_DIR=${BUILD_DIR:-build}
EXECUTABLE=${EXECUTABLE:-gemm}
# EXECUTABLE=${EXECUTABLE:-mpi-transform}
NUM_JOBS=${NUM_JOBS:-4}

set -e

cmake -B "${BUILD_DIR}" -S . -D CMAKE_BUILD_TYPE=Release -D CMAKE_EXPORT_COMPILE_COMMANDS=ON -D ENABLE_MPI=ON >&2
cmake --build "${BUILD_DIR}" --parallel "$(nproc)" >&2

mpirun -N "${NUM_JOBS}" --host="$(hostname):${NUM_JOBS}" "${BUILD_DIR}/${EXECUTABLE}"
