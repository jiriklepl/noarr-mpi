#!/bin/bash

set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

BUILD_DIR=$(realpath "${BUILD_DIR}")

CONFIG=${1:-Release}

git submodule update --init --recursive

(
	cd vendor/boost
	./bootstrap.sh
	echo "using mpi ;" > project-config.jam
	./b2 --prefix="${BUILD_DIR}/boost-install" \
		--with-mpi \
		--with-system \
		--with-serialization \
		--with-test \
		install
)

CMAKE_PREFIX_PATH="${BUILD_DIR}/boost-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"

# Configure Kokkos
cmake -B "${BUILD_DIR}/kokkos-build" -S vendor/kokkos \
	-G "Unix Makefiles" \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/kokkos-install" \
	-D Kokkos_ENABLE_OPENMP=OFF \
	-D Kokkos_ENABLE_SERIAL=ON \
	-D Kokkos_ARCH_NATIVE=ON \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build and install Kokkos
cmake --build "${BUILD_DIR}/kokkos-build" --target install --config "${CONFIG}" --parallel "${NUM_JOBS}"

CMAKE_PREFIX_PATH="${BUILD_DIR}/kokkos-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
export CMAKE_PREFIX_PATH

# Configure Kokkos-comm
cmake -B "${BUILD_DIR}/kokkos-comm-build" -S vendor/kokkos-comm \
	-G "Unix Makefiles" \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
	-D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/kokkos-comm-install" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build and install Kokkos-comm
cmake --build "${BUILD_DIR}/kokkos-comm-build" --target install --config "${CONFIG}" --parallel "${NUM_JOBS}"

CMAKE_PREFIX_PATH="${BUILD_DIR}/kokkos-comm-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
export CMAKE_PREFIX_PATH

cmake -B "${BUILD_DIR}" -S . \
	-G "Unix Makefiles" \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON
