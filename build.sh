#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

TARGET=${1:-all}
CONFIG=${2:-Release}

shift 2 || true

cmake --build "${BUILD_DIR}" --parallel "${NUM_JOBS}" --target "${TARGET}" --config "${CONFIG}" "$@"
