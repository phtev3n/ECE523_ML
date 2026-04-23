"""Compile individual shot overlay videos into a single demo reel.

Reads all seq_*_overlay.mp4 files from --in_dir and concatenates them
using OpenCV into a single demo_reel.mp4 in the same directory.

Usage
-----
python scripts/compile_demo_reel.py --in_dir outputs/demo_videos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate per-shot overlay videos into a demo reel")
    parser.add_argument("--in_dir", required=True, help="Directory containing seq_*_overlay.mp4 files")
    parser.add_argument("--out", default=None, help="Output path (default: <in_dir>/demo_reel.mp4)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    clips = sorted(in_dir.glob("seq_*_overlay.mp4"))

    if not clips:
        sys.exit(f"ERROR: No seq_*_overlay.mp4 files found in {in_dir}")

    print(f"Found {len(clips)} clips:")
    for c in clips:
        print(f"  {c.name}")

    # Read first clip to get video properties
    probe = cv2.VideoCapture(str(clips[0]))
    fps = probe.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    out_path = Path(args.out) if args.out else in_dir / "demo_reel.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    total_frames = 0
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            total_frames += 1
        cap.release()
        print(f"  appended {clip.name}")

    writer.release()
    print(f"\nDemo reel saved to {out_path}  ({total_frames} frames @ {fps:.0f} fps)")


if __name__ == "__main__":
    main()
