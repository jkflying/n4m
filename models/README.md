# Model Files

ONNX model files go here. They are gitignored.

## XFeat

Export with: `python tools/export/1_export_xfeat_onnx.py`

Produces `xfeat.onnx` — the XFeat backbone with raw outputs (descriptors, keypoint logits, reliability map). Postprocessing is done in C++.

## LightGlue

Not yet exported. See `tools/export/README.md` for details.
