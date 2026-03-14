#!/usr/bin/env python3
"""Test the exported LightGlue ONNX model against PyTorch reference."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from modules.lighterglue import LighterGlue


class LightGlueExportable(nn.Module):
    """Same wrapper used during export — needed for PyTorch reference."""

    def __init__(self, lighterglue: LighterGlue, filter_threshold: float = 0.1):
        super().__init__()
        self.net = lighterglue.net
        self.filter_threshold = filter_threshold
        self.net.conf.depth_confidence = -1
        self.net.conf.width_confidence = -1
        self.net.conf.flash = False

    def forward(self, kpts0, kpts1, desc0, desc1, image_size0, image_size1):
        size0 = image_size0.float()
        size1 = image_size1.float()
        kpts0 = self._normalize_keypoints(kpts0, size0)
        kpts1 = self._normalize_keypoints(kpts1, size1)

        desc0 = self.net.input_proj(desc0)
        desc1 = self.net.input_proj(desc1)

        encoding0 = self.net.posenc(kpts0)
        encoding1 = self.net.posenc(kpts1)

        for i in range(self.net.conf.n_layers):
            desc0, desc1 = self.net.transformers[i](desc0, desc1, encoding0, encoding1)

        scores, _ = self.net.log_assignment[i](desc0, desc1)
        m0, mscores0 = self._filter_matches(scores)
        return m0, mscores0

    @staticmethod
    def _normalize_keypoints(kpts, size):
        shift = size / 2
        scale = size.max(1, keepdim=True).values / 2
        return (kpts - shift[:, None]) / scale[:, None]

    def _filter_matches(self, scores):
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        m0 = max0.indices
        m1 = max1.indices

        indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
        mutual0 = indices0 == m1.gather(1, m0)

        max0_exp = max0.values.exp()
        zero = max0_exp.new_tensor(0)
        mscores0 = torch.where(mutual0, max0_exp, zero)

        valid0 = mutual0 & (mscores0 > self.filter_threshold)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        return m0, mscores0


def test_lightglue_onnx():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(root, 'models')

    # Load ONNX model
    session = ort.InferenceSession(os.path.join(models_dir, 'lightglue.onnx'))

    # Create test inputs
    M, N = 256, 300
    np.random.seed(42)
    kpts0 = np.random.rand(1, M, 2).astype(np.float32) * 640
    kpts1 = np.random.rand(1, N, 2).astype(np.float32) * 640
    desc0 = np.random.randn(1, M, 64).astype(np.float32)
    desc1 = np.random.randn(1, N, 64).astype(np.float32)
    image_size0 = np.array([[640, 480]], dtype=np.int64)
    image_size1 = np.array([[640, 480]], dtype=np.int64)

    # ONNX inference
    ort_outputs = session.run(None, {
        'kpts0': kpts0, 'kpts1': kpts1,
        'desc0': desc0, 'desc1': desc1,
        'image_size0': image_size0, 'image_size1': image_size1,
    })
    ort_matches, ort_scores = ort_outputs

    print(f"ONNX matches0: {ort_matches.shape}")
    print(f"ONNX scores0: {ort_scores.shape}")
    print(f"ONNX matched: {(ort_matches >= 0).sum()}")

    # PyTorch reference
    lighterglue = LighterGlue()
    lighterglue.eval()
    wrapper = LightGlueExportable(lighterglue, filter_threshold=0.1)
    wrapper.eval()

    with torch.no_grad():
        pt_matches, pt_scores = wrapper(
            torch.from_numpy(kpts0), torch.from_numpy(kpts1),
            torch.from_numpy(desc0), torch.from_numpy(desc1),
            torch.from_numpy(image_size0), torch.from_numpy(image_size1),
        )

    pt_matches = pt_matches.numpy()
    pt_scores = pt_scores.numpy()

    print(f"\nPyTorch matches0: {pt_matches.shape}")
    print(f"PyTorch scores0: {pt_scores.shape}")
    print(f"PyTorch matched: {(pt_matches >= 0).sum()}")

    match_diff = np.abs(pt_matches.astype(np.int64) - ort_matches.astype(np.int64))
    score_diff = np.abs(pt_scores - ort_scores)

    print(f"\nmatches0 max diff: {match_diff.max()}")
    print(f"scores0 mean abs diff: {score_diff.mean():.6f}")
    print(f"scores0 max abs diff: {score_diff.max():.6f}")

    assert match_diff.max() == 0, f"Match diff too large: {match_diff.max()}"
    assert score_diff.mean() < 1e-4, f"Score diff too large: {score_diff.mean()}"
    print("\nAll outputs match!")


if __name__ == '__main__':
    test_lightglue_onnx()
