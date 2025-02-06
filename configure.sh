#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}

cmake -B "${BUILD_DIR}" -S . -D CMAKE_BUILD_TYPE=Release -D CMAKE_EXPORT_COMPILE_COMMANDS=ON
