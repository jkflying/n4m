# Model Export Tools

Requires the `external/venv_models` virtualenv (created by `tools/install_dependencies.sh`)
and the `external/accelerated_features` checkout (clone from https://github.com/verlab/accelerated_features).

Run in order:

1. **`1_export_xfeat_onnx.py`** — Export XFeat backbone to ONNX format. Produces `models/xfeat.onnx`.
2. **`2_export_reference.py`** — Generate reference keypoints/descriptors from the Python XFeat implementation. Saves `.npy` files to `test/reference/` for C++ test verification.
3. **`3_test_onnx_model.py`** — Verify the exported ONNX model produces outputs matching the PyTorch reference.
4. **`4_export_lightglue_onnx.py`** — Export LightGlue matcher to ONNX format. Produces `models/lightglue.onnx`.
5. **`5_export_lightglue_reference.py`** — Generate reference match data from Python LightGlue for C++ test verification.
