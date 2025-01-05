#!/bin/bash

# This script runs clang-format on all files in the repository to ensure a consistent code style.

set -e

mkdir -p include src test
find include/ src/ test/ \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
	-exec clang-format -i -style=file {} +
