from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np


# ── PyAV helpers ───────────────────────────────────────────────────────────────

def _open_container(video_path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(video_path))
    except Exception as e:
        sys.exit(f"Could not open video {video_path}: {e}")


def _video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    for stream in container.streams:
        if stream.type == "video":
            return stream
    raise RuntimeError("No video stream found")


# ── fps detection ──────────────────────────────────────────────────────────────

def detect_fps(video_path: Path) -> float:
    """Read the frame rate of *video_path* using PyAV.

    Returns the detected fps as a float, rounded to the nearest standard
    rate (24, 25, 30, 48, 50, 60, 120, 240). Falls back to 60.0 on failure.
    """
    try:
        with _open_container(video_path) as container:
            stream = _video_stream(container)

            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            if rate is not None:
                raw = float(rate)
                for standard in (24, 25, 30, 48, 50, 60, 120, 240):
                    if abs(raw - standard) < 0.5:
                        return float(standard)
                return raw

            if stream.duration is not None and stream.time_base is not None and stream.frames:
                dur_s = float(stream.duration * stream.time_base)
                if dur_s > 0:
                    raw = float(stream.frames) / dur_s
                    for standard in (24, 25, 30, 48, 50, 60, 120, 240):
                        if abs(raw - standard) < 0.5:
                            return float(standard)
                    return raw

    except Exception:
        pass

    print("  WARNING: could not detect fps — defaulting to 60 fps")
    return 60.0


# ── frame extraction ───────────────────────────────────────────────────────────

def extract_frames_pyav(
    video_path: Path,
    start_frame: int,
    num_frames: int,
) -> list[np.ndarray]:
    """Decode *num_frames* starting at *start_frame* using PyAV.

    Returns a list of decoded BGR images as NumPy arrays.
    """
    if start_frame < 0:
        sys.exit("--impact_frame must be >= 0")
    if num_frames <= 0:
        sys.exit("--num_frames must be > 0")

    extracted: list[np.ndarray] = []

    with _open_container(video_path) as container:
        stream = _video_stream(container)

        # Seek near the target using AV_TIME_BASE (microseconds).
        # Do NOT pass stream= here — that would make the offset be interpreted in
        # the stream's time_base units (e.g. 1/90000), causing a large overshoot.
        if stream.average_rate is not None:
            try:
                target_seconds = max(0.0, (start_frame - 2) / float(stream.average_rate))
                container.seek(int(target_seconds * 1_000_000), backward=True, any_frame=False)
            except Exception:
                pass

        decoded_index = 0
        started = False

        for frame in container.decode(stream):
            # Prefer timestamp-derived index when possible.
            if frame.pts is not None and frame.time_base is not None and stream.average_rate is not None:
                try:
                    t = float(frame.pts * frame.time_base)
                    frame_index = int(round(t * float(stream.average_rate)))
                except Exception:
                    frame_index = decoded_index
            else:
                frame_index = decoded_index

            decoded_index += 1

            if frame_index < start_frame:
                continue

            started = True
            img = frame.to_ndarray(format="bgr24")
            extracted.append(img)

            if len(extracted) >= num_frames:
                break

        if not extracted:
            if not started:
                sys.exit(
                    "No frames were extracted. Check that --impact_frame is within "
                    "the video length."
                )
            sys.exit("Reached video end before extracting any usable frames.")

    return extracted


def centre_crop_square(img: np.ndarray) -> np.ndarray:
    """Centre-crop an image to the largest inscribed square."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def _orient_frame(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply clockwise rotation so the saved frame is upright."""
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def process_frames(raw_frames: list[np.ndarray], out_dir: Path, target_size: int = 256, orient: int = 0) -> int:
    """Centre-crop, rotate, and resize frames, saving to *out_dir*.

    Returns the number of frames successfully written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for i, img in enumerate(raw_frames):
        if img is None or img.size == 0:
            print(f"  WARNING: frame {i} is empty, skipping")
            continue

        img = centre_crop_square(img)
        img = _orient_frame(img, orient)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        ok = cv2.imwrite(str(out_dir / f"{i:06d}.png"), img)
        if not ok:
            print(f"  WARNING: could not write frame {i:06d}.png")
            continue

        written += 1

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and preprocess video frames for golf_tracer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
fps is auto-detected from the video file. Use --fps only to override.

Recommended --num_frames by fps:
  30 fps : 24 frames = 0.8 s window  (recommended — more trajectory context)
  60 fps : 24 frames = 0.4 s window  (default training config)
  60 fps : 48 frames = 0.8 s window  (retrain with sequence_length: 48)
        """,
    )
    parser.add_argument("--video", required=True, help="Path to .mov video file")
    parser.add_argument("--impact_frame", type=int, required=True,
                        help="Frame index of club-ball impact (0-based)")
    parser.add_argument("--num_frames", type=int, default=24,
                        help="Number of frames to extract starting at impact (default: 24)")
    parser.add_argument("--out_dir", required=True, help="Output directory for PNG frames")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override frame rate (auto-detected from video if omitted)")
    parser.add_argument("--size", type=int, default=256,
                        help="Target square image size in pixels (default: 256)")
    parser.add_argument("--orient", type=int, default=0,
                        help="Clockwise rotation to apply to each frame before saving: "
                             "0=none, 90=CW, 180=upside-down, 270=CCW (default: 0)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    if args.impact_frame < 0:
        sys.exit("--impact_frame must be >= 0")
    if args.num_frames <= 0:
        sys.exit("--num_frames must be > 0")
    if args.size <= 0:
        sys.exit("--size must be > 0")

    if args.fps is not None:
        fps = args.fps
        print(f"fps: {fps:.2f} (user override)")
    else:
        fps = detect_fps(video_path)
        print(f"fps: {fps:.2f} (auto-detected from metadata)")

    duration_s = args.impact_frame / fps + (args.num_frames / fps)
    print(
        f"Extracting {args.num_frames} frames from '{video_path.name}'\n"
        f"  Start frame : {args.impact_frame}  ({args.impact_frame / fps:.2f} s)\n"
        f"  Window      : {args.num_frames / fps:.2f} s\n"
        f"  End time    : {duration_s:.2f} s"
    )

    out_dir = Path(args.out_dir)

    raw_frames = extract_frames_pyav(
        video_path=video_path,
        start_frame=args.impact_frame,
        num_frames=args.num_frames,
    )
    n = process_frames(raw_frames, out_dir, args.size, orient=args.orient)

    if n == 0:
        sys.exit("No output frames were written.")

    print(f"Saved {n} frames to {out_dir}  ({args.size}x{args.size} px)")

    meta_path = out_dir.parent / "sequence_meta.json"
    meta = {
        "video": str(video_path),
        "fps": fps,
        "impact_frame": args.impact_frame,
        "num_frames": n,
        "image_size": args.size,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Sequence metadata saved to {meta_path}")


if __name__ == "__main__":
    main()