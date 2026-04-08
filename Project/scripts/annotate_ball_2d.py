"""Interactive GUI for annotating golf ball 2D positions in frame sequences.

Usage
-----
python scripts/annotate_ball_2d.py \
    --frames_dir real_dataset/seq_0000/frames \
    --out annotations_2d.json

Controls
--------
  Left-click   : mark ball center (u, v)
  'n'          : mark frame as NOT visible (visible=0) and advance
  'z'          : undo last annotation and go back one frame
  'q'          : quit and save progress
  Enter/Space  : confirm annotation and advance to next frame

The output JSON is a list of {frame_index, visible, uv} dicts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Globals for mouse callback
_click_uv: list[float] | None = None
_display_img: np.ndarray | None = None
WINDOW = "Annotate Ball — click ball center | 'n'=not visible | 'z'=undo | 'q'=quit"


def _mouse_callback(event, x, y, flags, param):
    global _click_uv, _display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_uv = [float(x), float(y)]
        # Draw crosshair on display copy
        if _display_img is not None:
            display = _display_img.copy()
            cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(WINDOW, display)


def annotate_sequence(frames_dir: Path, existing: list[dict] | None = None) -> list[dict]:
    """Open an interactive window to annotate ball positions."""
    global _click_uv, _display_img

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        sys.exit(f"No PNG frames in {frames_dir}")

    annotations = list(existing) if existing else []
    start_idx = len(annotations)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 768, 768)
    cv2.setMouseCallback(WINDOW, _mouse_callback)

    i = start_idx
    while i < len(frame_paths):
        img = cv2.imread(str(frame_paths[i]))
        if img is None:
            print(f"Could not read {frame_paths[i]}, skipping")
            i += 1
            continue

        # Show frame with zoom for visibility
        display = img.copy()
        h, w = display.shape[:2]
        label = f"Frame {i}/{len(frame_paths)-1}"
        cv2.putText(display, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # If revisiting a previously annotated frame, show old annotation
        if i < len(annotations) and annotations[i]["visible"] == 1:
            u, v = annotations[i]["uv"]
            cv2.drawMarker(display, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.putText(display, "(previous)", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        _display_img = display.copy()
        _click_uv = None
        cv2.imshow(WINDOW, display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return annotations

            if key == ord("n"):
                # Not visible
                if i < len(annotations):
                    annotations[i] = {"frame_index": i, "visible": 0, "uv": [0.0, 0.0]}
                else:
                    annotations.append({"frame_index": i, "visible": 0, "uv": [0.0, 0.0]})
                print(f"  frame {i}: NOT VISIBLE")
                i += 1
                break

            if key == ord("z"):
                # Undo
                if i > 0:
                    i -= 1
                    if len(annotations) > i:
                        annotations.pop()
                    print(f"  undo → back to frame {i}")
                break

            if key in (13, 32):  # Enter or Space
                if _click_uv is not None:
                    if i < len(annotations):
                        annotations[i] = {"frame_index": i, "visible": 1, "uv": _click_uv}
                    else:
                        annotations.append({"frame_index": i, "visible": 1, "uv": _click_uv})
                    print(f"  frame {i}: uv = [{_click_uv[0]:.1f}, {_click_uv[1]:.1f}]")
                    i += 1
                    break

    cv2.destroyAllWindows()
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate golf ball 2D positions in frame sequences")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing 256x256 PNG frames")
    parser.add_argument("--out", type=str, default="annotations_2d.json", help="Output JSON path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing annotation file")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        sys.exit(f"Frames directory not found: {frames_dir}")

    existing = None
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Resuming from {len(existing)} existing annotations")

    annotations = annotate_sequence(frames_dir, existing)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nSaved {len(annotations)} annotations to {out_path}")


if __name__ == "__main__":
    main()
