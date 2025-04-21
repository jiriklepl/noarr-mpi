#!/bin/bash

# This script runs clang-format on all files in the repository to ensure a consistent code style.

set -euo pipefail

mkdir -p include tests examples
find include/ tests/ examples/ \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
	-exec clang-format -i -style=file {} +
