#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}

CONFIG=${1:-Release}

shift || true

ctest --test-dir "${BUILD_DIR}" -C "${CONFIG}" "$@"
