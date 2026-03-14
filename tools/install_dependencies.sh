#! /bin/bash

set -ex
sudo apt-get update || (apt-get update && apt-get install -y sudo)
sudo apt-get install -y cmake ccache ninja-build libopencv-dev libgtest-dev libspdlog-dev clang-format-14 clang-tidy-18 clang-18 build-essential git lcov libonnxruntime-dev python3-venv curl

cd "$(dirname "$0")/.."

rm -rf external/venv/
python3 -m venv external/venv
external/venv/bin/pip install fastcov

git config --global --add safe.directory "$(pwd)"
git submodule update --init --recursive
