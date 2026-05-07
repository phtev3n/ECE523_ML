"""Generate individual pipeline stage images for presentation (seq_0006, 7-iron).

Outputs (in --out_dir):
  stage_01_raw_frame.png     — raw impact frame, no overlay
  stage_02_annotation.png    — GT ball positions annotated frame-by-frame
  stage_03_detection.png     — detector predictions on impact frame
  stage_04_tracking.png      — Kalman-filtered 2D trail mid-sequence
  stage_05_overlay.png       — extracted frame from final tracer overlay video

Usage
-----
python scripts/capture_pipeline_stages.py \
    --seq_dir     real_data_work/dataset/seq_0006 \
    --predictions outputs/real_test_results/seq_0006_predictions.json \
    --overlay_mp4 <path>/seq_0006_overlay.mp4 \
    --out_dir     outputs/pipeline_stages
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Presentation colour palette ────────────────────────────────────────────────
_CYAN   = (255, 212, 0)    # BGR — detector / Kalman
_ORANGE = (0, 140, 255)    # BGR — GT annotation
_YELLOW = (0, 255, 255)    # BGR — impact marker
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)

STAGE_FONT       = cv2.FONT_HERSHEY_SIMPLEX
STAGE_FONT_SCALE = 1.1
STAGE_THICKNESS  = 2


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_frame(seq_dir: Path, index: int) -> np.ndarray:
    path = seq_dir / "frames" / f"{index:06d}.png"
    img  = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Cannot read frame: {path}")
    return img


def _label_bar(img: np.ndarray, text: str, subtitle: str = "") -> np.ndarray:
    """Dark translucent bar at top with stage label."""
    out  = img.copy()
    h, w = out.shape[:2]
    bar_h = 70 if subtitle else 52
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
    cv2.putText(out, text, (18, 36), STAGE_FONT, STAGE_FONT_SCALE,
                _WHITE, STAGE_THICKNESS, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (18, 58), STAGE_FONT, 0.55,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return out


def _dot(img: np.ndarray, uv: tuple[float, float],
         color: tuple[int, int, int], radius: int = 10,
         filled: bool = True, thickness: int = -1) -> None:
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(img, (u, v), radius, color, thickness if not filled else -1,
               cv2.LINE_AA)


def _crosshair(img: np.ndarray, uv: tuple[float, float],
               color: tuple[int, int, int], arm: int = 22) -> None:
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.line(img, (u - arm, v), (u + arm, v), color, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - arm), (u, v + arm), color, 2, cv2.LINE_AA)


def _text_label(img: np.ndarray, uv: tuple[float, float], text: str,
                color: tuple[int, int, int], scale: float = 0.55) -> None:
    u, v = int(round(uv[0])) + 14, int(round(uv[1])) - 8
    (tw, th), bl = cv2.getTextSize(text, STAGE_FONT, scale, 1)
    cv2.rectangle(img, (u - 3, v - th - 3), (u + tw + 3, v + bl + 1),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (u, v), STAGE_FONT, scale, color, 1, cv2.LINE_AA)


def _bbox(img: np.ndarray, uv: tuple[float, float],
          half: int, color: tuple[int, int, int]) -> None:
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.rectangle(img, (u - half, v - half), (u + half, v + half),
                  color, 2, cv2.LINE_AA)


def _trail(img: np.ndarray,
           uvs: list[tuple[float, float]],
           color: tuple[int, int, int],
           max_radius: int = 8) -> None:
    """Draw fading dots from oldest (small/dim) to newest (large/bright)."""
    n = len(uvs)
    for i, uv in enumerate(uvs):
        alpha = (i + 1) / n          # 0→1 oldest→newest
        r     = max(3, int(max_radius * alpha))
        c     = tuple(int(ch * (0.35 + 0.65 * alpha)) for ch in color)
        _dot(img, uv, c, radius=r)
    if n >= 2:
        pts = np.array([[int(round(u)), int(round(v))] for u, v in uvs],
                       dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def crop_roi(img: np.ndarray, center_uv: tuple[float, float],
             pad: int = 260) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a square ROI around center_uv, return (crop, (x0, y0)) offset."""
    h, w = img.shape[:2]
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    x0 = max(0, u - pad)
    y0 = max(0, v - pad)
    x1 = min(w, u + pad)
    y1 = min(h, v + pad)
    return img[y0:y1, x0:x1], (x0, y0)


# ── Stage generators ───────────────────────────────────────────────────────────

def stage_01_raw(seq_dir: Path, out_dir: Path) -> None:
    img = load_frame(seq_dir, 0)
    img = _label_bar(img, "Stage 1 — Raw Impact Frame",
                     "Unprocessed iPhone 60 fps video frame at ball impact")
    cv2.imwrite(str(out_dir / "stage_01_raw_frame.png"), img)
    print("  stage_01_raw_frame.png")


def stage_02_annotation(seq_dir: Path, ann: dict, out_dir: Path) -> None:
    """Impact frame with GT annotations drawn for visible frames."""
    img    = load_frame(seq_dir, 0)
    frames = ann["frames"]

    # Draw trail of GT dots for all visible frames
    visible_uvs = [(fr["uv"][0], fr["uv"][1])
                   for fr in frames if fr["visible"]]
    _trail(img, visible_uvs, _ORANGE, max_radius=10)

    # Emphasise frame 0 with crosshair
    uv0 = (frames[0]["uv"][0], frames[0]["uv"][1])
    _crosshair(img, uv0, _ORANGE, arm=26)
    _dot(img, uv0, _ORANGE, radius=7)
    _text_label(img, uv0, f"GT  ({uv0[0]:.0f}, {uv0[1]:.0f})", _ORANGE)

    # Label count
    vis_count = sum(1 for fr in frames if fr["visible"])
    img = _label_bar(img, "Stage 2 — Manual Ball Annotation",
                     f"Click-annotated GT positions: {vis_count} visible / {len(frames)} frames")
    cv2.imwrite(str(out_dir / "stage_02_annotation.png"), img)
    print("  stage_02_annotation.png")


def stage_03_detection(seq_dir: Path, pred: dict, out_dir: Path) -> None:
    """Impact frame with raw detector output for all frames overlaid."""
    img = load_frame(seq_dir, 0)

    measured = pred["measured_uv"]
    vis_prob  = pred.get("visible_prob", [1.0] * len(measured))

    # Draw all detector hits as a trail
    uvs = [(m[0], m[1]) for m, p in zip(measured, vis_prob) if p > 0.5]
    _trail(img, uvs, _CYAN, max_radius=8)

    # Emphasise frame 0 detection
    uv0 = (measured[0][0], measured[0][1])
    _bbox(img, uv0, half=20, color=_CYAN)
    _dot(img, uv0, _CYAN, radius=5)
    _text_label(img, uv0, f"p={vis_prob[0]:.2f}", _CYAN)

    img = _label_bar(img, "Stage 3 — Multi-Scale Ball Detector",
                     "YOLOv5-style detector output; raw (unfiltered) predictions per frame")
    cv2.imwrite(str(out_dir / "stage_03_detection.png"), img)
    print("  stage_03_detection.png")


def stage_04_tracking(seq_dir: Path, pred: dict, out_dir: Path,
                      show_up_to_frame: int = 12) -> None:
    """Mid-sequence frame showing Kalman-filtered trail."""
    img = load_frame(seq_dir, show_up_to_frame)

    filtered = pred["filtered_uv"][:show_up_to_frame + 1]
    measured = pred["measured_uv"][:show_up_to_frame + 1]

    # Raw detections — dim orange dots
    for uv in measured:
        _dot(img, (uv[0], uv[1]), _ORANGE, radius=4)

    # Kalman trail — bright cyan
    uvs_filt = [(uv[0], uv[1]) for uv in filtered]
    _trail(img, uvs_filt, _CYAN, max_radius=10)

    # Current position marker
    cur = uvs_filt[-1]
    _crosshair(img, cur, _CYAN, arm=18)

    # Legend
    h, w = img.shape[:2]
    for i, (col, label) in enumerate(zip([_ORANGE, _CYAN],
                                         ["Raw detection", "Kalman filtered"])):
        oy = h - 55 + i * 26
        cv2.circle(img, (24, oy), 7, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (38, oy + 5), STAGE_FONT, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    img = _label_bar(img, "Stage 4 — Kalman Filter Tracking",
                     f"Smoothed 2D trajectory through frame {show_up_to_frame}")
    cv2.imwrite(str(out_dir / "stage_04_tracking.png"), img)
    print("  stage_04_tracking.png")


def stage_05_overlay(overlay_mp4: Path, out_dir: Path,
                     target_frame: int = 14) -> None:
    """Extract a representative frame from the final tracer overlay video."""
    cap = cv2.VideoCapture(str(overlay_mp4))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {overlay_mp4}, skipping stage 5")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = min(target_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()

    if not ok or img is None:
        print(f"  WARNING: could not read frame {idx} from overlay video")
        return

    img = _label_bar(img, "Stage 5 — Pipeline Tracer Overlay",
                     f"Detector + Kalman filter + tracer rendering  (frame {idx})")
    cv2.imwrite(str(out_dir / "stage_05_overlay.png"), img)
    print("  stage_05_overlay.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate individual pipeline stage images for presentation"
    )
    parser.add_argument("--seq_dir",
                        default="real_data_work/dataset/seq_0006",
                        help="Sequence directory (contains frames/ and annotations.json)")
    parser.add_argument("--predictions",
                        default="outputs/real_test_results/seq_0006_predictions.json",
                        help="Predictions JSON for this sequence")
    parser.add_argument("--overlay_mp4",
                        default=None,
                        help="Path to seq_0006_overlay.mp4 (for stage 5)")
    parser.add_argument("--out_dir",
                        default="outputs/pipeline_stages")
    parser.add_argument("--tracking_frame",
                        type=int, default=12,
                        help="Frame index to show in the Kalman tracking stage")
    parser.add_argument("--overlay_frame",
                        type=int, default=14,
                        help="Frame index to extract from the overlay video")
    args = parser.parse_args()

    seq_dir     = Path(args.seq_dir)
    pred_path   = Path(args.predictions)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not seq_dir.is_dir():
        sys.exit(f"Sequence directory not found: {seq_dir}")
    if not pred_path.exists():
        sys.exit(f"Predictions file not found: {pred_path}")

    with open(seq_dir / "annotations.json") as f:
        ann = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)

    print(f"Generating pipeline stage images -> {out_dir}")
    stage_01_raw(seq_dir, out_dir)
    stage_02_annotation(seq_dir, ann, out_dir)
    stage_03_detection(seq_dir, pred, out_dir)
    stage_04_tracking(seq_dir, pred, out_dir, args.tracking_frame)

    if args.overlay_mp4:
        stage_05_overlay(Path(args.overlay_mp4), out_dir, args.overlay_frame)
    else:
        print("  stage_05_overlay.png  (skipped — no --overlay_mp4 provided)")

    print(f"\nDone. Images saved to {out_dir}/")


if __name__ == "__main__":
    main()
