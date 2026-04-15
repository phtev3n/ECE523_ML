"""Generate a printable checkerboard calibration pattern as a PNG.

The output is sized for US Letter paper (8.5 x 11 in) at 300 DPI.
Each square is 30mm. The board has 9 columns x 7 rows of squares
(8 x 6 inner corners — what OpenCV counts as the "pattern size").

Usage
-----
python scripts/generate_checkerboard.py --out calibration_pattern.png
Then: print calibration_pattern.png at 100% scale (no fit-to-page).
Measure one printed square with a ruler and verify it is 30mm.
If not, note the actual size and pass --square_mm <measured> to calibrate_camera.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_checkerboard(
    cols: int = 9,       # total squares per row (inner corners = cols-1)
    rows: int = 7,       # total squares per column (inner corners = rows-1)
    square_mm: float = 30.0,
    dpi: int = 300,
    margin_mm: float = 15.0,
) -> np.ndarray:
    mm_per_inch = 25.4
    px_per_mm = dpi / mm_per_inch

    sq_px = int(round(square_mm * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    # US Letter at 300 DPI
    page_w_px = int(round(8.5 * dpi))
    page_h_px = int(round(11.0 * dpi))

    board_w = cols * sq_px
    board_h = rows * sq_px

    img = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255

    # Centre the board on the page
    off_x = (page_w_px - board_w) // 2
    off_y = (page_h_px - board_h) // 2

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = off_x + c * sq_px
                y0 = off_y + r * sq_px
                img[y0:y0 + sq_px, x0:x0 + sq_px] = 0

    # Label with key parameters at the bottom
    label = (f"Checkerboard  {cols}x{rows} squares  ({cols-1}x{rows-1} inner corners)  "
             f"square={square_mm:.0f}mm  print at 100% — verify square size before calibrating")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tx = (page_w_px - tw) // 2
    ty = off_y + board_h + margin_px // 2 + th
    cv2.putText(img, label, (tx, ty), font, scale, 0, thick, cv2.LINE_AA)

    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="calibration_pattern.png")
    parser.add_argument("--cols", type=int, default=9,
                        help="Total square columns (inner corners = cols-1, default 9→8)")
    parser.add_argument("--rows", type=int, default=7,
                        help="Total square rows (inner corners = rows-1, default 7→6)")
    parser.add_argument("--square_mm", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    img = make_checkerboard(args.cols, args.rows, args.square_mm, args.dpi)
    out = Path(args.out)
    cv2.imwrite(str(out), img)
    print(f"Saved {out}  ({img.shape[1]}x{img.shape[0]} px @ {args.dpi} DPI)")
    print(f"  Pattern: {args.cols}x{args.rows} squares  →  {args.cols-1}x{args.rows-1} inner corners")
    print(f"  Square size: {args.square_mm}mm")
    print(f"  IMPORTANT: print at exactly 100% scale (disable 'fit to page')")
    print(f"  After printing, measure a square with a ruler.")
    print(f"  If it differs from {args.square_mm}mm, use --square_mm <measured> in calibrate_camera.py")


if __name__ == "__main__":
    main()
