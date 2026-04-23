"""Compile individual shot overlay videos into a single demo reel.

Reads all seq_*_overlay.mp4 files from --in_dir and concatenates them
using ffmpeg into a single demo_reel.mp4 in the same directory.

Usage
-----
python scripts/compile_demo_reel.py --in_dir outputs/demo_videos
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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

    list_path = in_dir / "concat_list.txt"
    with open(list_path, "w") as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")

    out_path = Path(args.out) if args.out else in_dir / "demo_reel.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_path),
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit("ERROR: ffmpeg failed. Ensure ffmpeg is available on the HPC.")

    print(f"\nDemo reel saved to {out_path}")
    print("Individual clips are in:", in_dir)


if __name__ == "__main__":
    main()
