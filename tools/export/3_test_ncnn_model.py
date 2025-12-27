#!/usr/bin/env python3
"""Test the exported ncnn model against PyTorch reference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import numpy as np
import ncnn
import torch
import torch.nn.functional as F
from modules.model import XFeatModel


def test_ncnn_model():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(root, 'models')

    # Load ncnn model
    net = ncnn.Net()
    net.load_param(os.path.join(models_dir, 'xfeat.param'))
    net.load_model(os.path.join(models_dir, 'xfeat.bin'))

    # Create test input
    H, W = 480, 640
    np.random.seed(42)
    img = np.random.rand(H, W, 3).astype(np.float32)

    # ncnn inference
    mat_in = ncnn.Mat.from_pixels(
        (img * 255).astype(np.uint8), ncnn.Mat.PIXEL_RGB, W, H
    )
    # Normalize to [0,1] - but actually ncnn model expects raw float input
    # Let's use Mat directly
    mat_in = ncnn.Mat(img.transpose(2, 0, 1))  # CHW

    ex = net.create_extractor()
    ex.input("in0", mat_in)

    _, out0 = ex.extract("out0")  # descriptors
    _, out1 = ex.extract("out1")  # heatmap
    _, out2 = ex.extract("out2")  # reliability

    print(f"ncnn out0 (descriptors): {out0.numpy().shape if out0 is not None else 'None'}")
    print(f"ncnn out1 (heatmap): {out1.numpy().shape if out1 is not None else 'None'}")
    print(f"ncnn out2 (reliability): {out2.numpy().shape if out2 is not None else 'None'}")

    # PyTorch reference
    model = XFeatModel()
    weights_path = os.path.join(root, 'external', 'accelerated_features', 'weights', 'xfeat.pt')
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        M, K, Hout = model(x)
        M = F.normalize(M, dim=1)

    print(f"\nPyTorch M (descriptors): {M.shape}")
    print(f"PyTorch K (keypoint logits): {K.shape}")
    print(f"PyTorch H (reliability): {Hout.shape}")

    if out0 is not None:
        ncnn_desc = np.array(out0)
        torch_desc = M.numpy().squeeze()
        diff = np.abs(ncnn_desc - torch_desc).mean()
        print(f"\nDescriptor mean abs diff: {diff:.6f}")


if __name__ == '__main__':
    test_ncnn_model()
