#!/usr/bin/env python3
"""Export XFeat backbone to ncnn format via TorchScript + pnnx.

Exports only the backbone (XFeatModel), not the postprocessing.
Postprocessing (descriptor normalization, heatmap unfold, NMS) is done in C++.

Input: [1, 3, H, W] RGB float image (H,W divisible by 32)
Outputs:
  - out0: [1, 64, H/8, W/8] raw descriptors (NOT L2-normalized)
  - out1: [1, 65, H/8, W/8] keypoint logits (65 channels: 64 spatial bins + 1 dustbin)
  - out2: [1, 1, H/8, W/8] reliability map (sigmoid output)
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import XFeatModel


class XFeatExportable(XFeatModel):
    """XFeatModel with unfold replaced by pixel_unshuffle for ncnn compatibility."""

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

    # Export via TorchScript
    ts_path = os.path.join(models_dir, 'xfeat.pt')
    traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)
    print(f"Saved TorchScript to {ts_path}")

    # Run pnnx
    pnnx_bin = os.path.join(os.path.dirname(sys.executable), 'pnnx')
    if not os.path.exists(pnnx_bin):
        pnnx_bin = 'pnnx'

    cmd = [
        pnnx_bin, ts_path,
        f'inputshape=[1,3,{H},{W}]',
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=models_dir)

    # pnnx produces xfeat.ncnn.param and xfeat.ncnn.bin
    param_src = os.path.join(models_dir, 'xfeat.ncnn.param')
    bin_src = os.path.join(models_dir, 'xfeat.ncnn.bin')

    if os.path.exists(param_src) and os.path.exists(bin_src):
        param_dst = os.path.join(models_dir, 'xfeat.param')
        bin_dst = os.path.join(models_dir, 'xfeat.bin')
        os.replace(param_src, param_dst)
        os.replace(bin_src, bin_dst)
        print(f"ncnn model: {param_dst}, {bin_dst}")
    else:
        print("ERROR: pnnx did not produce expected output files")
        sys.exit(1)

    # Verify shapes
    with torch.no_grad():
        M, K, Hout = model(dummy)
    print(f"  descriptors (M): {M.shape}")
    print(f"  keypoint logits (K): {K.shape}")
    print(f"  reliability (H): {Hout.shape}")


if __name__ == '__main__':
    main()
