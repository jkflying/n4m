# Model Export Tools

Requires the `external/venv_models` virtualenv (created by `tools/install_dependencies.sh`)
and the `external/accelerated_features` checkout (clone from https://github.com/verlab/accelerated_features).

Run in order:

1. **`1_export_xfeat_ncnn.py`** — Export XFeat backbone to ncnn format via TorchScript + pnnx. Produces `models/xfeat.param` and `models/xfeat.bin`.
2. **`2_export_reference.py`** — Generate reference keypoints/descriptors from the Python XFeat implementation. Saves `.npy` files to `test/test_data/reference/` for C++ test verification.
3. **`3_test_ncnn_model.py`** — Verify the exported ncnn model produces outputs matching the PyTorch reference.
