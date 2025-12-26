#!/usr/bin/env python3
"""Export reference data from XFeat + LightGlue for C++ verification.

Usage:
    python export_reference.py <test_image1> <test_image2> [output_dir]

Outputs .npy files in output_dir (default: test/test_data/reference/).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def export_xfeat_reference(model, image_path: Path, output_dir: Path, prefix: str = "xfeat"):
    """Run XFeat and save intermediate + final outputs."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Save test image
    cv2.imwrite(str(output_dir / "test_image.jpg"), img_bgr)

    # Preprocess: BGR->RGB, resize to 640x480, normalize to [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 480))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    np.save(output_dir / f"{prefix}_input_tensor.npy", img_tensor.numpy())

    # Run model
    with torch.no_grad():
        output = model.detectAndCompute(img_bgr, top_k=4096)

    keypoints = output[0]  # Nx2
    scores = output[1]     # N
    descriptors = output[2]  # Nx64

    # Convert to numpy
    kpts_np = keypoints.cpu().numpy()
    scores_np = scores.cpu().numpy()
    desc_np = descriptors.cpu().numpy()

    # Save keypoints as Nx3 (x, y, score)
    kpts_with_scores = np.column_stack([kpts_np, scores_np])
    np.save(output_dir / f"{prefix}_keypoints.npy", kpts_with_scores.astype(np.float32))
    np.save(output_dir / f"{prefix}_descriptors.npy", desc_np.astype(np.float32))

    print(f"XFeat: {len(kpts_np)} keypoints from {image_path.name}")
    return keypoints, scores, descriptors


def export_lightglue_reference(model, kpts0, desc0, kpts1, desc1, output_dir: Path):
    """Run LightGlue and save intermediate + final outputs."""
    # Pack inputs as LightGlue expects
    data = {
        "keypoints0": kpts0.unsqueeze(0),
        "keypoints1": kpts1.unsqueeze(0),
        "descriptors0": desc0.unsqueeze(0),
        "descriptors1": desc1.unsqueeze(0),
    }

    np.save(output_dir / "lg_input_kpts0.npy", kpts0.cpu().numpy().astype(np.float32))
    np.save(output_dir / "lg_input_kpts1.npy", kpts1.cpu().numpy().astype(np.float32))
    np.save(output_dir / "lg_input_desc0.npy", desc0.cpu().numpy().astype(np.float32))
    np.save(output_dir / "lg_input_desc1.npy", desc1.cpu().numpy().astype(np.float32))

    with torch.no_grad():
        result = model(data)

    matches = result["matches"]  # Mx2
    confidence = result["matching_scores"]  # M

    matches_np = matches.cpu().numpy()
    confidence_np = confidence.cpu().numpy()

    np.save(output_dir / "lg_matches.npy", matches_np.astype(np.int32))
    np.save(output_dir / "lg_confidence.npy", confidence_np.astype(np.float32))

    print(f"LightGlue: {len(matches_np)} matches")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("test/test_data/reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import XFeat
    try:
        from modules.xfeat import XFeat
    except ImportError:
        print("Error: XFeat not found. Clone https://github.com/verlab/accelerated_features")
        print("and run from within that directory, or add it to PYTHONPATH.")
        sys.exit(1)

    # Import LightGlue
    try:
        from lightglue import LightGlue as LightGlueModel
    except ImportError:
        print("Error: LightGlue not found. Install via: pip install lightglue")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # XFeat
    xfeat = XFeat()
    xfeat = xfeat.to(device).eval()

    kpts0, scores0, desc0 = export_xfeat_reference(xfeat, img1_path, output_dir)
    kpts1, scores1, desc1 = export_xfeat_reference(xfeat, img2_path, output_dir, prefix="xfeat2")

    # LightGlue
    lightglue = LightGlueModel(features="xfeat").to(device).eval()
    export_lightglue_reference(lightglue, kpts0, desc0, kpts1, desc1, output_dir)

    print(f"\nReference data saved to {output_dir}/")


if __name__ == "__main__":
    main()
