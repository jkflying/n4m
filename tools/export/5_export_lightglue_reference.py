#!/usr/bin/env python3
"""Export reference LightGlue match data for C++ test verification.

Runs XFeat + LightGlue on two real test images and saves the matches
as .npy files for comparison in C++ tests.

Outputs (to test/reference/):
  - lightglue_matches0.npy:  [M] int64 match indices
  - lightglue_scores0.npy:   [M] float32 match scores
  - lightglue_num_kpts0.npy: scalar — number of keypoints in image 0
  - lightglue_num_kpts1.npy: scalar — number of keypoints in image 1
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import cv2
import numpy as np
import torch
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue


def load_and_resize(path, max_dim=1600):
    """Load image and resize so longest side is max_dim."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, 'test', 'reference')
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(root, 'test', 'image_test_data')
    img0_path = os.path.join(image_dir, 'P2530253.JPG')
    img1_path = os.path.join(image_dir, 'P2540254.JPG')

    img0 = load_and_resize(img0_path, max_dim=1600)
    img1 = load_and_resize(img1_path, max_dim=1600)
    print(f"Image 0: {img0.shape[1]}x{img0.shape[0]}")
    print(f"Image 1: {img1.shape[1]}x{img1.shape[0]}")

    # Extract features with XFeat
    xfeat = XFeat(top_k=4096)
    feats0 = xfeat.detectAndCompute(img0, top_k=4096)[0]
    feats1 = xfeat.detectAndCompute(img1, top_k=4096)[0]

    # XFeat returns unbatched tensors [N, D] — add batch dim for LightGlue
    kpts0 = feats0['keypoints'].unsqueeze(0) if feats0['keypoints'].dim() == 2 else feats0['keypoints']
    kpts1 = feats1['keypoints'].unsqueeze(0) if feats1['keypoints'].dim() == 2 else feats1['keypoints']
    desc0 = feats0['descriptors'].unsqueeze(0) if feats0['descriptors'].dim() == 2 else feats0['descriptors']
    desc1 = feats1['descriptors'].unsqueeze(0) if feats1['descriptors'].dim() == 2 else feats1['descriptors']

    n0 = kpts0.shape[1]
    n1 = kpts1.shape[1]
    print(f"Keypoints: {n0} + {n1}")

    # Match with LightGlue
    lighterglue = LighterGlue()
    lighterglue.eval()

    data = {
        'keypoints0': kpts0,
        'descriptors0': desc0,
        'image_size0': torch.tensor([[img0.shape[1], img0.shape[0]]]),  # (W, H)
        'keypoints1': kpts1,
        'descriptors1': desc1,
        'image_size1': torch.tensor([[img1.shape[1], img1.shape[0]]]),  # (W, H)
    }

    with torch.no_grad():
        result = lighterglue.forward(data, min_conf=0.0)

    matches0 = result['matches0'][0].cpu().numpy()     # [M]
    scores0 = result['matching_scores0'][0].cpu().numpy()  # [M]

    num_matched = (matches0 >= 0).sum()
    print(f"Matches: {num_matched}")
    print(f"Score range: [{scores0[matches0 >= 0].min():.4f}, {scores0[matches0 >= 0].max():.4f}]")

    # Save
    np.save(os.path.join(output_dir, 'lightglue_matches0.npy'), matches0.astype(np.int64))
    np.save(os.path.join(output_dir, 'lightglue_scores0.npy'), scores0.astype(np.float32))
    np.save(os.path.join(output_dir, 'lightglue_num_kpts0.npy'), np.array(n0, dtype=np.int64))
    np.save(os.path.join(output_dir, 'lightglue_num_kpts1.npy'), np.array(n1, dtype=np.int64))

    print(f"\nSaved reference data to {output_dir}")


if __name__ == '__main__':
    main()
