"""Extract, center-crop, and resize video frames for the golf_tracer pipeline.

Usage
-----
python scripts/extract_frames.py \
    --video IMG_0001.MOV \
    --impact_frame 47 \
    --num_frames 24 \
    --out_dir real_dataset/seq_0000/frames
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def extract_frames_ffmpeg(video_path: Path, tmp_dir: Path, start_frame: int, num_frames: int, fps: float = 60.0) -> list[Path]:
    """Use ffmpeg to extract a range of frames from *video_path*."""
    start_time = start_frame / fps
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.6f}",
        "-i", str(video_path),
        "-frames:v", str(num_frames),
        "-start_number", "0",
        str(tmp_dir / "%06d.png"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(tmp_dir.glob("*.png"))


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """Center-crop an image to the largest inscribed square."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def process_frames(raw_paths: list[Path], out_dir: Path, target_size: int = 256) -> None:
    """Center-crop and resize frames, saving to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(raw_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"WARNING: could not read {path}, skipping")
            continue
        img = center_crop_square(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{i:06d}.png"), img)
    print(f"Saved {len(raw_paths)} frames to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and preprocess video frames for golf_tracer")
    parser.add_argument("--video", type=str, required=True, help="Path to iPhone .MOV video file")
    parser.add_argument("--impact_frame", type=int, required=True, help="Frame number of club-ball impact (0-indexed)")
    parser.add_argument("--num_frames", type=int, default=24, help="Number of frames to extract starting at impact")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed PNG frames")
    parser.add_argument("--fps", type=float, default=60.0, help="Video frame rate")
    parser.add_argument("--size", type=int, default=256, help="Target square image size")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    with tempfile.TemporaryDirectory() as tmp:
        raw_paths = extract_frames_ffmpeg(video_path, Path(tmp), args.impact_frame, args.num_frames, args.fps)
        if not raw_paths:
            sys.exit("ffmpeg produced no frames — check video path and impact_frame")
        process_frames(raw_paths, Path(args.out_dir), args.size)


if __name__ == "__main__":
    main()
