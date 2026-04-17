"""Diagnostic: try to detect checkerboard corners in a calibration image.

Saves a resized preview PNG so you can verify the image loaded correctly,
then brute-forces a range of pattern sizes to find one that works.

Usage
-----
python scripts/debug_calibration.py calibration_images/IMG_9720.HEIC
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _imread_any(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  HEIC load failed: {e}")
        return None


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is None or not path.exists():
        # pick first image in calibration_images/
        candidates = sorted(Path("calibration_images").glob("*"))
        if not candidates:
            sys.exit("No image provided and calibration_images/ is empty")
        path = candidates[0]

    print(f"Loading: {path}")
    img = _imread_any(path)
    if img is None:
        sys.exit("Could not load image — check pillow-heif installation")

    h, w = img.shape[:2]
    print(f"  Loaded OK: {w}x{h} px")

    # Save a small preview
    preview_path = "debug_preview.png"
    scale = min(1.0, 800 / max(w, h))
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imwrite(preview_path, small)
    print(f"  Saved preview → {preview_path}  (open this to verify image looks correct)")

    # Try all reasonable inner-corner counts
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found_any = False
    for cols_inner in range(4, 13):
        for rows_inner in range(3, 10):
            pattern = (cols_inner, rows_inner)
            ok, corners = cv2.findChessboardCorners(gray, pattern, None)
            if ok:
                print(f"\n  FOUND corners with inner pattern {cols_inner}x{rows_inner}")
                print(f"  → use --cols {cols_inner + 1} --rows {rows_inner + 1} in calibrate_camera.py")
                # Draw and save
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern, corners, ok)
                vis_small = cv2.resize(vis, (int(w * scale), int(h * scale)))
                cv2.imwrite("debug_corners.png", vis_small)
                print(f"  Saved corner overlay → debug_corners.png")
                found_any = True
                break
        if found_any:
            break

    if not found_any:
        print("\n  No pattern found for any inner corner count 4–12 x 3–9.")
        print("  Possible causes:")
        print("    1. Image is blurry, overexposed, or the board is not fully in frame")
        print("    2. The board was photographed on a screen (Moire/glare prevents detection)")
        print("    3. The board is partially occluded or not flat")
        print("  Open debug_preview.png and confirm the checkerboard is clearly visible.")


if __name__ == "__main__":
    main()
