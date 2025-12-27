#!/usr/bin/env python3
"""Test the exported ONNX model against PyTorch reference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from modules.model import XFeatModel


def test_onnx_model():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(root, 'models')

    # Load ONNX model
    session = ort.InferenceSession(os.path.join(models_dir, 'xfeat.onnx'))

    # Create test input
    H, W = 480, 640
    np.random.seed(42)
    img = np.random.rand(1, 3, H, W).astype(np.float32)

    # ONNX inference
    ort_outputs = session.run(None, {'input': img})
    ort_desc, ort_logits, ort_rel = ort_outputs

    print(f"ONNX descriptors: {ort_desc.shape}")
    print(f"ONNX keypoint_logits: {ort_logits.shape}")
    print(f"ONNX reliability: {ort_rel.shape}")

    # PyTorch reference
    model = XFeatModel()
    weights_path = os.path.join(root, 'external', 'accelerated_features', 'weights', 'xfeat.pt')
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    x = torch.from_numpy(img)
    with torch.no_grad():
        M, K, Hout = model(x)

    print(f"\nPyTorch descriptors: {M.shape}")
    print(f"PyTorch keypoint_logits: {K.shape}")
    print(f"PyTorch reliability: {Hout.shape}")

    desc_diff = np.abs(ort_desc - M.numpy()).mean()
    logits_diff = np.abs(ort_logits - K.numpy()).mean()
    rel_diff = np.abs(ort_rel - Hout.numpy()).mean()

    print(f"\nDescriptor mean abs diff: {desc_diff:.6f}")
    print(f"Logits mean abs diff: {logits_diff:.6f}")
    print(f"Reliability mean abs diff: {rel_diff:.6f}")

    assert desc_diff < 1e-5, f"Descriptor diff too large: {desc_diff}"
    assert logits_diff < 1e-5, f"Logits diff too large: {logits_diff}"
    assert rel_diff < 1e-5, f"Reliability diff too large: {rel_diff}"
    print("\nAll outputs match!")


if __name__ == '__main__':
    test_onnx_model()
