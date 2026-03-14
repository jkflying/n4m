![CI Status](https://github.com/jkflying/n4m/workflows/C/C++%20CI/badge.svg)
[![codecov](https://codecov.io/gh/jkflying/n4m/branch/master/graph/badge.svg)](https://codecov.io/gh/jkflying/n4m)

# n4m
## Neural Network Nearest Neighbour Matching
A C++17 library for local feature extraction and matching using XFeat and LightGlue, backed by ONNX Runtime.

Extracts keypoints and descriptors with XFeat, then matches them across image pairs with LightGlue. Ships with optional grid-based spatial distribution (cell NMS) and a CLI tool for visualizing matches.

Dependencies: OpenCV, ONNX Runtime, spdlog. Designed for use with opencalibration, or really anything else
