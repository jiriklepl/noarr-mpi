#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}

CONFIG=${1:-Release}

cmake -B "${BUILD_DIR}" -S . -D CMAKE_BUILD_TYPE="${CONFIG}" -D CMAKE_EXPORT_COMPILE_COMMANDS=ON
