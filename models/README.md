# Model Files

ONNX model files go here. They are gitignored.

## XFeat

Export with: `python tools/export/1_export_xfeat_onnx.py`

Produces `xfeat.onnx` — the XFeat backbone with raw outputs (descriptors, keypoint logits, reliability map). Postprocessing is done in C++.

## LightGlue

Export with: `python tools/export/4_export_lightglue_onnx.py`

Produces `lightglue.onnx` — the LightGlue matcher (XFeat variant, 6 layers, 1 head, 96-dim descriptors projected from 64-dim input). Early stopping and point pruning are disabled for a static ONNX graph.

Inputs: keypoints, descriptors, and image sizes from both images.
Outputs: `matches0 [1,M]` (match index or -1) and `scores0 [1,M]` (confidence).
