#! /bin/bash

set -ex
sudo apt-get update || (apt-get update && apt-get install sudo)
sudo apt-get install -y cmake ccache ninja-build libopencv-dev libgtest-dev libspdlog-dev clang-format-14 clang-tidy-18 clang-18 build-essential git libvulkan-dev lcov libprotobuf-dev protobuf-compiler

cd "$(dirname "$0")/.."

rm -rf external/venv/
python3 -m venv external/venv
external/venv/bin/pip install fastcov

# Build ncnn from submodule

git submodule update --init --recursive

mkdir -p external/ncnn/build
cd external/ncnn/build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../install \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_VULKAN=OFF
ninja
ninja install
