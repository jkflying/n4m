#!/usr/bin/env python3
"""Export XFeat backbone to ONNX format.

Exports the backbone (XFeatModel) with raw outputs.
Postprocessing (heatmap unfold, descriptor normalization, NMS) is done in C++.

Input: [1, 3, H, W] RGB float image (H,W divisible by 32)
Outputs:
  - descriptors: [1, 64, H/8, W/8] raw descriptors (NOT L2-normalized)
  - keypoint_logits: [1, 65, H/8, W/8] keypoint logits
  - reliability: [1, 1, H/8, W/8] reliability map
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import torch
import torch.nn.functional as F
from modules.model import XFeatModel


class XFeatExportable(XFeatModel):
    """XFeatModel with unfold replaced by pixel_unshuffle for ONNX compatibility."""

    def _unfold2d(self, x, ws=2):
        return F.pixel_unshuffle(x, ws)


def main():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    weights_path = os.path.join(root, 'external', 'accelerated_features', 'weights', 'xfeat.pt')

    model = XFeatExportable()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    H, W = 480, 640
    dummy = torch.randn(1, 3, H, W)

    onnx_path = os.path.join(models_dir, 'xfeat.onnx')
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['descriptors', 'keypoint_logits', 'reliability'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'descriptors': {2: 'h8', 3: 'w8'},
            'keypoint_logits': {2: 'h8', 3: 'w8'},
            'reliability': {2: 'h8', 3: 'w8'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {onnx_path}")

    # Verify shapes
    with torch.no_grad():
        M, K, Hout = model(dummy)
    print(f"  descriptors (M): {M.shape}")
    print(f"  keypoint logits (K): {K.shape}")
    print(f"  reliability (H): {Hout.shape}")


if __name__ == '__main__':
    main()
