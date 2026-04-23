"""Assemble extracted frames, 2D annotations, and (optionally) 3D trajectories
into the dataset format expected by RealGolfSequenceDataset.

Supports two modes:
  --trajectories provided  : full 3D ground truth (from simulate_trajectory.py
                              or reconstruct_3d_from_2d.py)
  --trajectories omitted   : 2D-only mode for detector training (xyz set to zeros)

Usage
-----
# Full 3D mode (Trackman or video-reconstructed trajectories):
python scripts/build_dataset.py \
    --shot_map shot_map.json \
    --trajectories trajectories.json \
    --camera_params camera_params.json \
    --out_dir real_dataset

# 2D-only mode (no launch monitor, detector training only):
python scripts/build_dataset.py \
    --shot_map shot_map.json \
    --camera_params camera_params.json \
    --out_dir real_dataset \
    --mode_2d_only

shot_map.json
-------------
A list of per-shot entries:

[
  {
    "video": "IMG_0001.MOV",
    "impact_frame": 47,
    "frames_dir": "tmp/seq_0000/frames",
    "annotations_2d": "tmp/seq_0000/annotations_2d.json",
    "trajectory_index": 0
  },
  ...
]

The "trajectory_index" field is only needed when --trajectories is provided.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np


def reproject_xyz(xyz: np.ndarray, camera: dict) -> np.ndarray:
    """Project 3D points to 2D using the pipeline's projection convention.

    Matches frames extracted with --orient 180 (180° rotated pinhole):
      u_rot = (W-1-cx) - fx*x/z
      v_rot = (H-1-cy) + fy*(y - cam_h)/z
    Both u and v increase in the opposite direction from the standard pinhole.
    """
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    camera_height = float(camera.get("camera_height_m", 0.0))
    H = float(camera.get("image_h", 512))
    W = float(camera.get("image_w", 512))
    z = np.clip(xyz[:, 2], 1e-3, None)
    u = (W - 1 - cx) - fx * xyz[:, 0] / z
    v = (H - 1 - cy) + fy * (xyz[:, 1] - camera_height) / z
    return np.stack([u, v], axis=1)


def compute_reprojection_error(uv_annotated: np.ndarray, uv_reprojected: np.ndarray, visible: np.ndarray) -> dict:
    """Compute reprojection error stats on visible frames only."""
    mask = visible > 0.5
    if not np.any(mask):
        return {"mean_px": None, "max_px": None, "n_visible": 0}
    diff = uv_annotated[mask] - uv_reprojected[mask]
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return {
        "mean_px": float(np.mean(dist)),
        "max_px": float(np.max(dist)),
        "n_visible": int(np.sum(mask)),
    }


def build_one_sequence(
    seq_idx: int,
    entry: dict,
    trajectories: list[dict] | None,
    camera: dict,
    out_dir: Path,
    fps: float,
    reproj_threshold: float,
    mode_2d_only: bool,
) -> dict | None:
    """Build one sequence directory and annotations.json."""
    frames_src = Path(entry["frames_dir"])
    ann_2d_path = Path(entry["annotations_2d"])

    if not frames_src.is_dir():
        print(f"  seq {seq_idx}: frames dir not found: {frames_src}, skipping")
        return None
    if not ann_2d_path.exists():
        print(f"  seq {seq_idx}: 2D annotations not found: {ann_2d_path}, skipping")
        return None

    with open(ann_2d_path) as f:
        ann_2d = json.load(f)

    num_frames = len(ann_2d)

    # Load or generate 3D data
    traj_data = None
    if not mode_2d_only and trajectories is not None:
        traj_idx = entry.get("trajectory_index")
        if traj_idx is None or traj_idx >= len(trajectories):
            print(f"  seq {seq_idx}: trajectory index {traj_idx} out of range, skipping")
            return None
        traj_data = trajectories[traj_idx]
        xyz_list = traj_data["xyz"]
        num_frames = min(num_frames, len(xyz_list))
        xyz = np.array(xyz_list[:num_frames], dtype=np.float32)
    else:
        # 2D-only mode: fill xyz with zeros (detector training doesn't use xyz)
        xyz = np.zeros((num_frames, 3), dtype=np.float32)

    if num_frames == 0:
        print(f"  seq {seq_idx}: no frames, skipping")
        return None

    uv_ann = np.array([a["uv"] for a in ann_2d[:num_frames]], dtype=np.float32)
    visible = np.array([a["visible"] for a in ann_2d[:num_frames]], dtype=np.float32)

    # Reprojection validation (only meaningful with real 3D data)
    reproj = {"mean_px": None, "max_px": None, "n_visible": 0}
    if not mode_2d_only and traj_data is not None:
        uv_reproj = reproject_xyz(xyz, camera)
        reproj = compute_reprojection_error(uv_ann, uv_reproj, visible)
        status = f"seq {seq_idx}: {num_frames} frames, reproj mean={reproj['mean_px']}"
        if reproj["mean_px"] is not None and reproj["mean_px"] > reproj_threshold:
            print(f"  WARNING {status} > threshold {reproj_threshold}px — FLAGGED")
        else:
            print(f"  {status}")
    else:
        print(f"  seq {seq_idx}: {num_frames} frames (2D-only, xyz=zeros)")

    # Write sequence directory
    seq_name = f"seq_{seq_idx:04d}"
    seq_dir = out_dir / seq_name
    frames_out = seq_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    # Copy frames
    src_frames = sorted(frames_src.glob("*.png"))[:num_frames]
    for i, src in enumerate(src_frames):
        shutil.copy2(src, frames_out / f"{i:06d}.png")

    # Build annotations.json
    frame_records = []
    for i in range(num_frames):
        frame_records.append({
            "frame_index": i,
            "visible": int(visible[i]),
            "uv": uv_ann[i].tolist(),
            "xyz": xyz[i].tolist(),
        })

    annotation = {
        "fps": fps,
        "camera": camera,
        "frames": frame_records,
    }
    with open(seq_dir / "annotations.json", "w") as f:
        json.dump(annotation, f, indent=2)

    result = {
        "seq_name": seq_name,
        "num_frames": num_frames,
        "reprojection": reproj,
        "mode": "2d_only" if mode_2d_only else "3d",
    }
    if traj_data is not None:
        result["Cd"] = traj_data.get("Cd")
        result["Cm"] = traj_data.get("Cm")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build golf_tracer real dataset from processed components")
    parser.add_argument("--shot_map", type=str, required=True, help="JSON mapping shots to their components")
    parser.add_argument("--trajectories", type=str, default=None, help="Path to trajectories.json (from simulate_trajectory.py or reconstruct_3d_from_2d.py)")
    parser.add_argument("--camera_params", type=str, required=True, help="Path to camera_params.json from calibrate_camera.py")
    parser.add_argument("--out_dir", type=str, default="real_dataset", help="Output dataset root directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Video frame rate")
    parser.add_argument("--reproj_threshold", type=float, default=5.0, help="Reprojection error threshold in pixels to flag bad shots")
    parser.add_argument("--mode_2d_only", action="store_true", help="Build 2D-only dataset for detector training (no 3D trajectories needed)")
    args = parser.parse_args()

    if not args.mode_2d_only and args.trajectories is None:
        sys.exit("ERROR: --trajectories is required unless --mode_2d_only is set.\n"
                 "  For 2D detector training only:  add --mode_2d_only\n"
                 "  For full 3D pipeline:           provide --trajectories trajectories.json")

    # Load inputs
    with open(args.shot_map) as f:
        shot_map = json.load(f)
    # Normalize Windows backslash paths so they resolve on Linux
    for entry in shot_map:
        for key in ("frames_dir", "annotations_2d", "video"):
            if key in entry and isinstance(entry[key], str):
                entry[key] = entry[key].replace("\\", "/")
    trajectories = None
    if args.trajectories is not None:
        with open(args.trajectories) as f:
            trajectories = json.load(f)
    with open(args.camera_params) as f:
        cam_data = json.load(f)

    camera = cam_data["pipeline_intrinsics"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "2D-only (detector)" if args.mode_2d_only else "full 3D"
    print(f"Building dataset ({mode_label}): {len(shot_map)} shots -> {out_dir}")
    print(f"Camera: fx={camera['fx']:.1f} fy={camera['fy']:.1f} cx={camera['cx']:.1f} cy={camera['cy']:.1f} h={camera['camera_height_m']}m")

    results = []
    for i, entry in enumerate(shot_map):
        r = build_one_sequence(i, entry, trajectories, camera, out_dir, args.fps,
                               args.reproj_threshold, args.mode_2d_only)
        if r is not None:
            results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"Built {len(results)} / {len(shot_map)} sequences")
    reproj_means = [r["reprojection"]["mean_px"] for r in results if r["reprojection"]["mean_px"] is not None]
    if reproj_means:
        print(f"Reprojection error: mean={np.mean(reproj_means):.2f}px  max={np.max(reproj_means):.2f}px")
        flagged = sum(1 for v in reproj_means if v > args.reproj_threshold)
        if flagged:
            print(f"  {flagged} sequences exceed {args.reproj_threshold}px threshold — review these")

    # Save build report
    report_path = out_dir / "build_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Build report saved to {report_path}")


if __name__ == "__main__":
    main()
