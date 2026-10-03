#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
check_build=$(mktemp -d "${TMPDIR:-/tmp}/autograd-check.XXXXXX")
trap 'rm -rf "$check_build"' EXIT
compiler=${CXX:-g++}
flags=(-std=c++17 -Wall -Wextra -Werror -pedantic -I include)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
    flags+=(-O2)
fi
"$compiler" --version
for source in tests/scalar_test.cpp tests/mlp_test.cpp examples/scalar_demo.cpp examples/xor_demo.cpp; do
    executable="$check_build/$(basename "${source%.cpp}")"
    "$compiler" "${flags[@]}" "$source" -o "$executable"
    "$executable"
done
