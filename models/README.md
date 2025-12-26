# Model Conversion: ONNX → ncnn

## Prerequisites

```bash
pip install onnx onnxsim
# Build ncnn tools (onnx2ncnn) from https://github.com/Tencent/ncnn
```

## XFeat

1. Export ONNX from the XFeat repo:
   ```python
   # Using https://github.com/verlab/accelerated_features
   import torch
   from modules.xfeat import XFeat
   model = XFeat()
   model.eval()
   dummy = torch.randn(1, 3, 480, 640)
   torch.onnx.export(model, dummy, "xfeat.onnx",
                     input_names=["input"],
                     output_names=["heatmap", "descriptors"],
                     opset_version=12)
   ```

2. Simplify and convert:
   ```bash
   onnxsim xfeat.onnx xfeat_sim.onnx
   onnx2ncnn xfeat_sim.onnx xfeat.param xfeat.bin
   ```

3. Place `xfeat.param` and `xfeat.bin` in this directory.

## LightGlue

1. Export ONNX:
   ```python
   # Using https://github.com/fabio-sim/LightGlue-ONNX
   # Follow their export script for XFeat-compatible LightGlue
   python export.py --extractor_type xfeat --dynamic
   ```

2. Simplify (LightGlue has dynamic shapes, may need iteration):
   ```bash
   onnxsim lightglue.onnx lightglue_sim.onnx \
     --overwrite-input-shape "kpts0:1,512,2" "kpts1:1,512,2" \
     "desc0:1,512,64" "desc1:1,512,64"
   ```

3. Convert:
   ```bash
   onnx2ncnn lightglue_sim.onnx lightglue.param lightglue.bin
   ```

4. Place `lightglue.param` and `lightglue.bin` in this directory.

## Notes

- LightGlue uses attention layers with dynamic shapes. You may need to fix the input
  shapes for ncnn compatibility or use custom ops.
- If `onnx2ncnn` fails on LightGlue, try older opset versions or the
  [ncnn PNNX](https://github.com/pnnx/pnnx) tool as an alternative.
- For Vulkan support, ensure ncnn was built with `-DNCNN_VULKAN=ON`.
