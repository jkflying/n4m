#! /bin/bash

set -ex
sudo apt-get update || (apt-get update && apt-get install sudo)
sudo apt-get install -y cmake ccache ninja-build libopencv-dev libgtest-dev libspdlog-dev clang-format-14 clang-tidy-18 clang-18 build-essential git libvulkan-dev lcov libonnxruntime-dev

cd "$(dirname "$0")/.."

rm -rf external/venv/
python3 -m venv external/venv
external/venv/bin/pip install fastcov

git submodule update --init --recursive
