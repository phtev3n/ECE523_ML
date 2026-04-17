"""Camera intrinsic calibration from checkerboard images.

Usage
-----
python scripts/calibrate_camera.py \
    --image_dir calibration_images/ \
    --rows 6 --cols 8 --square_mm 30 \
    --camera_height_m 1.3 \
    --out camera_params.json

The output JSON contains fx, fy, cx, cy at the native image resolution,
scaled versions for 256x256 (assuming center-crop to square first), and
distortion coefficients.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _imread_any(path: Path) -> np.ndarray | None:
    """Read an image regardless of format, with HEIC support via pillow-heif."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    # cv2 cannot read HEIC — try pillow-heif as a fallback
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  could not read {path.name}: {e}")
        return None


def find_corners(image_paths: list[Path], pattern_size: tuple[int, int], square_mm: float):
    """Detect checkerboard corners in a set of images."""
    obj_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_mm

    obj_points = []
    img_points = []
    img_size = None

    for p in image_paths:
        img = _imread_any(p)
        if img is None:
            print(f"  skipping unreadable: {p.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(obj_point)
            img_points.append(corners)
            print(f"  corners found: {p.name}")
        else:
            print(f"  no corners:    {p.name}")

    return obj_points, img_points, img_size


def calibrate(obj_points, img_points, img_size):
    """Run OpenCV camera calibration and return intrinsics + distortion."""
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    print(f"\nCalibration RMS reprojection error: {ret:.4f} pixels")
    return K, dist


def scale_intrinsics(fx, fy, cx, cy, native_w, native_h, target_size=256):
    """Scale intrinsics from native resolution to target_size x target_size after center-crop to square."""
    side = min(native_w, native_h)
    # shift principal point for the center-crop
    cx_crop = cx - (native_w - side) / 2.0
    cy_crop = cy - (native_h - side) / 2.0
    s = target_size / side
    return {
        "fx": float(fx * s),
        "fy": float(fy * s),
        "cx": float(cx_crop * s),
        "cy": float(cy_crop * s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from checkerboard images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of checkerboard images")
    parser.add_argument("--rows", type=int, default=6, help="Inner corner rows on checkerboard")
    parser.add_argument("--cols", type=int, default=8, help="Inner corner columns on checkerboard")
    parser.add_argument("--square_mm", type=float, default=30.0, help="Checkerboard square size in mm")
    parser.add_argument("--camera_height_m", type=float, default=1.3, help="Camera height above ground in meters")
    parser.add_argument("--target_size", type=int, default=256, help="Target square image size for the pipeline")
    parser.add_argument("--out", type=str, default="camera_params.json", help="Output JSON path")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        sys.exit(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".heic", ".heif"}
    )
    if len(image_paths) < 3:
        sys.exit(f"Need at least 3 checkerboard images, found {len(image_paths)}")

    print(f"Found {len(image_paths)} images in {image_dir}")
    pattern_size = (args.cols - 1, args.rows - 1)  # inner corners
    obj_points, img_points, img_size = find_corners(image_paths, pattern_size, args.square_mm)

    if len(obj_points) < 3:
        sys.exit(f"Only {len(obj_points)} images had detected corners — need at least 3")

    K, dist = calibrate(obj_points, img_points, img_size)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    native_w, native_h = img_size

    scaled = scale_intrinsics(fx, fy, cx, cy, native_w, native_h, args.target_size)

    result = {
        "native_resolution": {"width": native_w, "height": native_h},
        "native_intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "distortion_coefficients": dist.flatten().tolist(),
        "pipeline_intrinsics": {
            **scaled,
            "camera_height_m": args.camera_height_m,
        },
        "target_size": args.target_size,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved calibration to {out_path}")
    print(f"  Native: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"  Pipeline ({args.target_size}x{args.target_size}): fx={scaled['fx']:.1f} fy={scaled['fy']:.1f} cx={scaled['cx']:.1f} cy={scaled['cy']:.1f}")


if __name__ == "__main__":
    main()
