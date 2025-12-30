#!/usr/bin/env python3
"""Export LightGlue (XFeat variant) to ONNX format.

Wraps the kornia LightGlue model with flat tensor inputs/outputs suitable for
ONNX export. Disables early stopping, point pruning, and flash attention so the
graph is static and exportable.

Inputs:
  - kpts0:       [1, M, 2] float32 keypoints from image 0
  - kpts1:       [1, N, 2] float32 keypoints from image 1
  - desc0:       [1, M, 64] float32 descriptors from image 0
  - desc1:       [1, N, 64] float32 descriptors from image 1
  - image_size0: [1, 2] int64 (height, width) of image 0
  - image_size1: [1, 2] int64 (height, width) of image 1

Outputs:
  - matches0: [1, M] int64 — index into image1 keypoints, -1 if unmatched
  - scores0:  [1, M] float32 — matching confidence scores
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.lighterglue import LighterGlue


class LightGlueExportable(nn.Module):
    """ONNX-exportable wrapper around LighterGlue.

    Takes flat tensor inputs, runs the LightGlue forward pass with no early
    stopping or point pruning, and returns matches0 + scores0.
    """

    def __init__(self, lighterglue: LighterGlue, filter_threshold: float = 0.1):
        super().__init__()
        self.net = lighterglue.net
        self.filter_threshold = filter_threshold

        # Disable dynamic behaviors for static ONNX graph
        self.net.conf.depth_confidence = -1   # no early stopping
        self.net.conf.width_confidence = -1   # no point pruning
        self.net.conf.flash = False           # no flash attention

    def forward(self, kpts0, kpts1, desc0, desc1, image_size0, image_size1):
        # kpts0: [1, M, 2], kpts1: [1, N, 2]
        # desc0: [1, M, 64], desc1: [1, N, 64]
        # image_size0: [1, 2], image_size1: [1, 2] (height, width)

        # Normalize keypoints
        size0 = image_size0.float()
        size1 = image_size1.float()
        kpts0 = self._normalize_keypoints(kpts0, size0)
        kpts1 = self._normalize_keypoints(kpts1, size1)

        # Project descriptors from input_dim (64) to descriptor_dim (96)
        desc0 = self.net.input_proj(desc0)
        desc1 = self.net.input_proj(desc1)

        # Positional encodings
        encoding0 = self.net.posenc(kpts0)
        encoding1 = self.net.posenc(kpts1)

        # Run all transformer layers (no early stopping, no pruning)
        for i in range(self.net.conf.n_layers):
            desc0, desc1 = self.net.transformers[i](desc0, desc1, encoding0, encoding1)

        # Compute assignment matrix from final layer
        scores, _ = self.net.log_assignment[i](desc0, desc1)

        # Filter matches
        m0, mscores0 = self._filter_matches(scores)

        return m0, mscores0

    @staticmethod
    def _normalize_keypoints(kpts, size):
        """Normalize keypoints to [-1, 1] range."""
        shift = size / 2
        scale = size.max(1, keepdim=True).values / 2
        return (kpts - shift[:, None]) / scale[:, None]

    def _filter_matches(self, scores):
        """Extract matches from log assignment matrix [1, M+1, N+1]."""
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        m0 = max0.indices  # [1, M]
        m1 = max1.indices  # [1, N]

        # Mutual nearest neighbor check
        indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
        mutual0 = indices0 == m1.gather(1, m0)

        # Scores
        max0_exp = max0.values.exp()
        zero = max0_exp.new_tensor(0)
        mscores0 = torch.where(mutual0, max0_exp, zero)

        # Apply threshold
        valid0 = mutual0 & (mscores0 > self.filter_threshold)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))

        return m0, mscores0


def main():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("Loading LighterGlue weights...")
    lighterglue = LighterGlue()
    lighterglue.eval()

    wrapper = LightGlueExportable(lighterglue, filter_threshold=0.1)
    wrapper.eval()
    wrapper.to('cpu')

    # Dummy inputs for tracing
    M, N = 512, 512
    kpts0 = torch.rand(1, M, 2) * 640
    kpts1 = torch.rand(1, N, 2) * 640
    desc0 = torch.randn(1, M, 64)
    desc1 = torch.randn(1, N, 64)
    image_size0 = torch.tensor([[640, 480]], dtype=torch.int64)  # (width, height)
    image_size1 = torch.tensor([[640, 480]], dtype=torch.int64)  # (width, height)

    # Verify PyTorch output
    with torch.no_grad():
        matches0_pt, scores0_pt = wrapper(kpts0, kpts1, desc0, desc1, image_size0, image_size1)
    print(f"PyTorch output: matches0 {matches0_pt.shape}, scores0 {scores0_pt.shape}")
    num_matched = (matches0_pt >= 0).sum().item()
    print(f"  {num_matched} matches out of {M} keypoints")

    # Export to ONNX
    onnx_path = os.path.join(models_dir, 'lightglue.onnx')
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        wrapper,
        (kpts0, kpts1, desc0, desc1, image_size0, image_size1),
        onnx_path,
        input_names=['kpts0', 'kpts1', 'desc0', 'desc1', 'image_size0', 'image_size1'],
        output_names=['matches0', 'scores0'],
        dynamic_axes={
            'kpts0': {1: 'num_kpts0'},
            'kpts1': {1: 'num_kpts1'},
            'desc0': {1: 'num_kpts0'},
            'desc1': {1: 'num_kpts1'},
            'matches0': {1: 'num_kpts0'},
            'scores0': {1: 'num_kpts0'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {onnx_path}")

    # Verify ONNX output matches PyTorch
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(onnx_path)
    onnx_inputs = {
        'kpts0': kpts0.numpy(),
        'kpts1': kpts1.numpy(),
        'desc0': desc0.numpy(),
        'desc1': desc1.numpy(),
        'image_size0': image_size0.numpy(),
        'image_size1': image_size1.numpy(),
    }
    matches0_onnx, scores0_onnx = sess.run(None, onnx_inputs)

    match_diff = np.abs(matches0_pt.numpy().astype(np.int64) - matches0_onnx.astype(np.int64))
    score_diff = np.abs(scores0_pt.numpy() - scores0_onnx)
    print(f"\nVerification:")
    print(f"  matches0 max diff: {match_diff.max()}")
    print(f"  scores0 mean abs diff: {score_diff.mean():.6f}")
    print(f"  scores0 max abs diff: {score_diff.max():.6f}")

    if match_diff.max() == 0 and score_diff.mean() < 1e-4:
        print("\n  ONNX output matches PyTorch output!")
    else:
        print("\n  WARNING: ONNX output differs from PyTorch!")

    # Try onnxsim if available
    try:
        import onnxsim
        import onnx
        print("\nSimplifying with onnxsim...")
        model = onnx.load(onnx_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, onnx_path)
            print(f"Simplified model saved to {onnx_path}")
        else:
            print("onnxsim simplification check failed, keeping original")
    except ImportError:
        print("\nonnxsim not available, skipping simplification")


if __name__ == '__main__':
    main()
