# n4m
A C++17 library for local feature extraction and matching using XFeat and LightGlue, backed by ONNX Runtime.

Extracts keypoints and descriptors with XFeat, then matches them across image pairs with LightGlue. Ships with optional grid-based spatial distribution (cell NMS) and a CLI tool for visualizing matches.

Dependencies: OpenCV, ONNX Runtime, spdlog.
