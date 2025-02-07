#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

TARGET=${1:-all}

cmake --build "${BUILD_DIR}" --parallel "${NUM_JOBS}" --target "${TARGET}"
