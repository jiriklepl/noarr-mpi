#!/bin/bash

set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build}

BUILD_DIR=$(realpath "${BUILD_DIR}")

CONFIG=${1:-Release}

git submodule update --init --recursive

# Configure Kokkos-comm
cmake -B "${BUILD_DIR}/kokkos-comm-build" -S vendor/kokkos-comm \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/kokkos-comm-install" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build and install Kokkos-comm
cmake --build "${BUILD_DIR}/kokkos-comm-build" --target install --config "${CONFIG}" --parallel

CMAKE_PREFIX_PATH="${BUILD_DIR}/kokkos-comm-install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

cmake -B "${BUILD_DIR}" -S . \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON
