#!/bin/bash

set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

DATASET=${DATASET:-"EXTRALARGE"}
case "${DATASET}" in
    ("EXTRALARGE")
        ;;
    ("LARGE")
        ;;
    ("MEDIUM")
		;;
	("SMALL")
		;;
	("MINI")
		;;
	(*)
		echo "Unknown dataset: ${DATASET}" >&2
		echo "Valid options are: EXTRALARGE, LARGE, MEDIUM, SMALL, MINI" >&2
		exit 1
esac

DATATYPE=${DATATYPE:-"FLOAT"}
case "${DATATYPE}" in
	("FLOAT")
		;;
	("DOUBLE")
		;;
	("INT")
		;;
	(*)
		echo "Unknown datatype: ${DATATYPE}" >&2
		echo "Valid options are: FLOAT, DOUBLE, INT" >&2
		exit 1
esac

BUILD_DIR=$(realpath "${BUILD_DIR}")

CONFIG=${1:-Release}

git submodule update --init --recursive

# Configure Kokkos
cmake -B "${BUILD_DIR}/kokkos-build" -S vendor/kokkos \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/kokkos-install" \
	-D Kokkos_ENABLE_OPENMP=ON \
	-D Kokkos_ARCH_NATIVE=ON \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build and install Kokkos
cmake --build "${BUILD_DIR}/kokkos-build" --target install --config "${CONFIG}" --parallel "${NUM_JOBS}"

CMAKE_PREFIX_PATH="${BUILD_DIR}/kokkos-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"

# Configure Kokkos-comm
cmake -B "${BUILD_DIR}/kokkos-comm-build" -S vendor/kokkos-comm \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
	-D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/kokkos-comm-install" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build and install Kokkos-comm
cmake --build "${BUILD_DIR}/kokkos-comm-build" --target install --config "${CONFIG}" --parallel "${NUM_JOBS}"

CMAKE_PREFIX_PATH="${BUILD_DIR}/kokkos-comm-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"

cmake -B "${BUILD_DIR}" -S . \
	-D CMAKE_BUILD_TYPE="${CONFIG}" \
	-D DATASET="${DATASET}" \
	-D DATATYPE="${DATATYPE}" \
	-D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON
