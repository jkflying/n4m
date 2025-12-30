#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

group()    { echo "::group::$1"; date; }
endgroup() { echo "::endgroup::"; }

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    group "Install dependencies"
    tools/install_dependencies.sh
    endgroup
fi

group "Check formatting"
tools/check_style.sh
endgroup

group "CMake (Debug)"
mkdir -p build_debug
cmake -S . -B build_debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
endgroup

group "Debug Build"
ninja -C build_debug
endgroup

group "Debug Test"
ninja -C build_debug test
endgroup

group "CMake (asan)"
mkdir -p build_asan
cmake -S . -B build_asan -G Ninja -DCMAKE_BUILD_TYPE=asan
endgroup

group "Address Sanitizer Build"
ninja -C build_asan
endgroup

group "Address Sanitizer Test"
ASAN_OPTIONS=detect_leaks=0 ninja -C build_asan test
endgroup

group "Coverage Build"
mkdir -p build_coverage
cmake -S . -B build_coverage -G Ninja -DCMAKE_BUILD_TYPE=Coverage
ninja -C build_coverage
endgroup

group "Coverage Test"
ninja -C build_coverage test_coverage_html
endgroup

group "Release Build + Package"
mkdir -p build_release
cmake -S . -B build_release -G Ninja -DCMAKE_BUILD_TYPE=Release -DNM_TESTING=OFF
ninja -C build_release
ninja -C build_release package
echo "Package: $(ls build_release/nnmatch-*.tar.gz)"
endgroup

echo "All CI steps passed."
