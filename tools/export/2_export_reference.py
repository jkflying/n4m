#!/usr/bin/env python3
"""Export reference data from XFeat for C++ verification.

Usage:
    python export_reference.py [output_dir]

Creates a synthetic test image and saves XFeat outputs as .npy files.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'accelerated_features'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from modules.xfeat import XFeat


def main():
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, 'test', 'test_data', 'reference')
    os.makedirs(output_dir, exist_ok=True)

    # Create a deterministic synthetic test image
    np.random.seed(42)
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), 2)
    cv2.circle(image, (400, 200), 50, (0, 0, 0), 3)
    cv2.imwrite(os.path.join(output_dir, 'test_image.png'), image)

    xfeat = XFeat(top_k=4096)

    # Run XFeat
    output = xfeat.detectAndCompute(image, top_k=4096)[0]
    keypoints = output['keypoints'].cpu().numpy()   # Nx2
    scores = output['scores'].cpu().numpy()          # N
    descriptors = output['descriptors'].cpu().numpy() # Nx64

    # Save as npy
    np.save(os.path.join(output_dir, 'xfeat_keypoints.npy'),
            np.column_stack([keypoints, scores]).astype(np.float32))
    np.save(os.path.join(output_dir, 'xfeat_descriptors.npy'),
            descriptors.astype(np.float32))

    print(f"Exported {len(keypoints)} keypoints to {output_dir}")
    print(f"  keypoints shape: {keypoints.shape}")
    print(f"  descriptors shape: {descriptors.shape}")
    print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")


if __name__ == '__main__':
    main()
