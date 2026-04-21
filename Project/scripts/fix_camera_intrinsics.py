"""Propagate corrected camera intrinsics to all seq_XXXX/annotations.json files.

The original camera_params.json was produced by a bad checkerboard calibration
(native cy=92.5 on a 3024-px tall frame, should be ~1512).  build_dataset.py
copied those wrong values into every seq_XXXX/annotations.json.  This script
replaces the camera block in every annotations.json with the corrected values
from camera_params.json, then re-runs reprojection error diagnostics.

Usage
-----
python scripts/fix_camera_intrinsics.py \
    --camera_params real_data_work/camera_params.json \
    --dataset_dir   real_data_work/dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def reproject_xyz(xyz: np.ndarray, camera: dict) -> np.ndarray:
    """Rotated-pinhole reprojection matching extract_frames.py --orient 180."""
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    cam_h = float(camera.get("camera_height_m", 0.0))
    H = float(camera.get("image_h", 512))
    W = float(camera.get("image_w", 512))
    z = np.clip(xyz[:, 2], 1e-3, None)
    u = (W - 1 - cx) - fx * xyz[:, 0] / z
    v = (H - 1 - cy) + fy * (xyz[:, 1] - cam_h) / z
    return np.stack([u, v], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch corrected camera intrinsics into dataset annotations")
    parser.add_argument("--camera_params", required=True, help="Path to corrected camera_params.json")
    parser.add_argument("--dataset_dir",   required=True, help="Root of real_data_work/dataset (contains seq_XXXX/)")
    args = parser.parse_args()

    cam_path = Path(args.camera_params)
    ds_path  = Path(args.dataset_dir)

    if not cam_path.exists():
        sys.exit(f"camera_params.json not found: {cam_path}")
    if not ds_path.is_dir():
        sys.exit(f"dataset_dir not found: {ds_path}")

    with open(cam_path) as f:
        cam_data = json.load(f)
    new_camera = cam_data["pipeline_intrinsics"]

    seq_dirs = sorted(ds_path.glob("seq_*"))
    if not seq_dirs:
        sys.exit(f"No seq_XXXX directories found under {ds_path}")

    print(f"Patching {len(seq_dirs)} sequences with corrected camera:")
    for k, v in new_camera.items():
        print(f"  {k}: {v}")
    print()

    reproj_errors = []

    for seq_dir in seq_dirs:
        ann_path = seq_dir / "annotations.json"
        if not ann_path.exists():
            print(f"  {seq_dir.name}: missing annotations.json, skipping")
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        old_camera = ann.get("camera", {})
        old_cy = old_camera.get("cy", "?")

        # Merge: keep image_h/image_w if present in old camera
        merged_camera = dict(new_camera)
        for extra_key in ("image_h", "image_w"):
            if extra_key in old_camera:
                merged_camera[extra_key] = old_camera[extra_key]

        ann["camera"] = merged_camera

        # Diagnostic: reprojection error before and after (only if xyz is non-zero)
        frames = ann.get("frames", [])
        xyz_all = np.array([f["xyz"] for f in frames], dtype=np.float32)
        uv_all  = np.array([f["uv"]  for f in frames], dtype=np.float32)
        vis_all = np.array([f.get("visible", 1) for f in frames], dtype=np.float32)
        mask    = (vis_all > 0.5) & (np.abs(xyz_all).sum(axis=1) > 0)

        if mask.any():
            uv_rep = reproject_xyz(xyz_all[mask], merged_camera)
            errs   = np.linalg.norm(uv_all[mask] - uv_rep, axis=1)
            mean_e = float(np.mean(errs))
            reproj_errors.append(mean_e)
            status = f"reproj_mean={mean_e:.1f}px  (old cy={old_cy})"
        else:
            status = "xyz=zeros (2D-only mode)"

        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        print(f"  {seq_dir.name}: {status}")

    print()
    if reproj_errors:
        print(f"Reprojection errors after fix:")
        print(f"  mean={np.mean(reproj_errors):.1f}px  max={np.max(reproj_errors):.1f}px")
        print("  (High values indicate the 3D XYZ labels from reconstruct_3d_from_2d.py")
        print("   need to be recomputed with the corrected intrinsics.)")
    print(f"\nPatched {len(seq_dirs)} sequences in {ds_path}")
    print("Next step: re-run reconstruct_3d_from_2d.py then build_dataset.py to regenerate valid XYZ labels.")


if __name__ == "__main__":
    main()
