#!/bin/bash

set -e

BUILD_DIR=${BUILD_DIR:-build}

ctest --test-dir "${BUILD_DIR}"
