from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
from golf_tracer.data.synthetic_dataset import export_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-built synthetic demo dataset with golfer silhouettes."
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_sequences", type=int, default=200)
    parser.add_argument("--sequence_length", type=int, default=24)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[512, 512],
        metavar=("H", "W"),
        help="Output frame resolution (H W). Must match the pipeline image_size "
             "in detector.yaml (default: 512 512). The old demo_dataset was generated "
             "at 256 256 — regenerate at 512 512 to avoid a silent size mismatch.",
    )
    args = parser.parse_args()
    export_synthetic_dataset(
        out_dir=args.out_dir,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        image_size=tuple(args.image_size),
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
