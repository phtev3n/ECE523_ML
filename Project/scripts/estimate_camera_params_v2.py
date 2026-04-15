from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from pymediainfo import MediaInfo


# ── 35mm full-frame sensor reference dimensions ───────────────────────────────
FRAME_35MM_W = 36.0   # mm
FRAME_35MM_H = 24.0   # mm

# ── iPhone 15 Pro Max lens map: 35mm-equiv focal length → lens label ─────────
IPHONE_15_PRO_MAX_LENSES = {
    13:  "0.5x ultrawide",
    24:  "1x wide (main)",
    48:  "2x telephoto crop",
    120: "5x periscope tele",
}

FL_MATCH_TOLERANCE = 2.0

# ── Per-model fallback table when metadata extraction fails ───────────────────
IPHONE_FALLBACK_FL: Dict[str, float] = {
    "iphone_15_pro": 24.0,
    "iphone_15": 26.0,
    "iphone_14_pro": 24.0,
    "iphone_14": 26.0,
    "iphone_13_pro": 26.0,
    "iphone_13": 26.0,
    "iphone_12_pro": 26.0,
    "iphone_12": 26.0,
    "default": 26.0,
}


# ── MediaInfo / exiftool helpers ───────────────────────────────────────────────

def _run_mediainfo(video_path: Path) -> Dict[str, Any]:
    """
    Return a ffprobe-like metadata dictionary using pymediainfo.
    """
    try:
        media_info = MediaInfo.parse(str(video_path))
    except Exception as e:
        sys.exit(f"MediaInfo failed on {video_path}: {e}")

    video_track = None
    general_track = None

    for track in media_info.tracks:
        if track.track_type == "General" and general_track is None:
            general_track = track
        elif track.track_type == "Video" and video_track is None:
            video_track = track

    if video_track is None:
        sys.exit(f"No video stream found in {video_path}")

    video_data = video_track.to_data() if hasattr(video_track, "to_data") else {}
    general_data = general_track.to_data() if (general_track and hasattr(general_track, "to_data")) else {}

    return {
        "streams": [
            {
                "codec_type": "video",
                "width": _safe_int(video_data.get("width")),
                "height": _safe_int(video_data.get("height")),
                "r_frame_rate": _extract_frame_rate_string(video_data),
                "avg_frame_rate": _extract_frame_rate_string(video_data),
                "tags": {k: v for k, v in video_data.items() if v is not None},
            }
        ],
        "format": {
            "tags": {k: v for k, v in general_data.items() if v is not None}
        }
    }


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, str):
            cleaned = "".join(ch for ch in val if ch.isdigit())
            return int(cleaned) if cleaned else None
        return int(val)
    except (ValueError, TypeError):
        return None


def _extract_frame_rate_string(video_data: Dict[str, Any]) -> str:
    """
    Return a simple fps string consumable by _parse_fps.
    """
    candidates = [
        video_data.get("frame_rate"),
        video_data.get("frame_rate_nominal"),
        video_data.get("other_frame_rate"),
    ]

    for c in candidates:
        if c is None:
            continue

        if isinstance(c, list):
            for item in c:
                parsed = _extract_numeric_prefix(item)
                if parsed is not None:
                    return str(parsed)

        parsed = _extract_numeric_prefix(c)
        if parsed is not None:
            return str(parsed)

    return "60"


def _extract_numeric_prefix(val) -> Optional[float]:
    try:
        s = str(val).strip()
        cleaned = []
        dot_seen = False
        for ch in s:
            if ch.isdigit():
                cleaned.append(ch)
            elif ch == "." and not dot_seen:
                cleaned.append(ch)
                dot_seen = True
            else:
                if cleaned:
                    break
        if not cleaned:
            return None
        return float("".join(cleaned))
    except (ValueError, TypeError):
        return None


def _try_exiftool_fl(video_path: Path) -> Optional[float]:
    """
    Return FocalLengthIn35mmFilm via exiftool, or None.
    """
    try:
        r = subprocess.run(
            ["exiftool", "-FocalLengthIn35mmFilm", "-json", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout)
        if data:
            val = data[0].get("FocalLengthIn35mmFilm")
            if val is not None:
                f = float(str(val).replace("mm", "").strip())
                if 10 < f < 200:
                    return f
    except (FileNotFoundError, json.JSONDecodeError, ValueError,
            subprocess.TimeoutExpired):
        pass
    return None


# ── Core metadata extraction ───────────────────────────────────────────────────

def extract_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Extract resolution, fps, and focal length from video metadata.

    Tries metadata sources in this order:
      1. pymediainfo video/general tags
      2. exiftool FocalLengthIn35mmFilm
      3. user-supplied --f35mm_override
      4. built-in fallback
    """
    meta = _run_mediainfo(video_path)

    video_stream = next(
        (s for s in meta.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        sys.exit(f"No video stream found in {video_path}")

    width = video_stream.get("width")
    height = video_stream.get("height")
    if width is None or height is None:
        sys.exit(f"Could not read width/height from video metadata: {video_path}")

    fps = _parse_fps(
        video_stream.get("r_frame_rate")
        or video_stream.get("avg_frame_rate")
        or meta.get("format", {}).get("tags", {}).get("com.apple.quicktime.framerate")
        or "60"
    )

    f35mm, fl_source = _extract_focal_length(video_stream, meta)
    if f35mm is None:
        exif_f = _try_exiftool_fl(video_path)
        if exif_f is not None:
            f35mm, fl_source = exif_f, "exiftool: FocalLengthIn35mmFilm"

    lens_label = _match_lens(f35mm)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "f35mm": f35mm,
        "fl_source": fl_source,
        "lens_label": lens_label,
    }


def _parse_fps(fps_str: str) -> float:
    """
    Parse fps from a rational string like '60000/1001' or a float-like string.
    """
    try:
        parts = str(fps_str).split("/")
        if len(parts) == 2:
            val = float(parts[0]) / float(parts[1])
        else:
            val = float(parts[0])

        for standard in (24, 25, 30, 48, 50, 60, 120, 240):
            if abs(val - standard) < 0.5:
                return float(standard)
        return val
    except (ValueError, ZeroDivisionError):
        return 60.0


def _extract_focal_length(video_stream: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Search MediaInfo-derived tag dicts for a focal length value.
    """
    tag_dicts = [
        (video_stream.get("tags", {}), "stream tag"),
        (meta.get("format", {}).get("tags", {}), "format tag"),
    ]

    focal_keys = [
        "com.apple.quicktime.camera.focal_length_35mm_equivalent",
        "com.apple.quicktime.focal_length_35mm",
        "focallengthin35mmfilm",
        "focal_length_35mm",
        "focal_length",
        "focallength",
        "focal length in 35mm format",
        "focal_length_in_35mm_format",
    ]

    for tags, source_label in tag_dicts:
        lower_tags = {str(k).lower(): v for k, v in tags.items()}

        for key in focal_keys:
            val = lower_tags.get(key)
            if val is not None:
                parsed = _parse_focal_value(val)
                if parsed is not None:
                    return parsed, f"{source_label}: {key}"

        for key, val in lower_tags.items():
            if "focal" in key:
                parsed = _parse_focal_value(val)
                if parsed is not None:
                    return parsed, f"{source_label}: {key}"

    return None, "not found"


def _parse_focal_value(val) -> Optional[float]:
    try:
        s = str(val).lower().replace("mm", "").strip()
        token = []
        dot_seen = False
        for ch in s:
            if ch.isdigit():
                token.append(ch)
            elif ch == "." and not dot_seen:
                token.append(ch)
                dot_seen = True
            elif token:
                break
        if not token:
            return None
        f = float("".join(token))
        if 10 < f < 200:
            return f
    except (ValueError, TypeError):
        pass
    return None


def _match_lens(f35mm: Optional[float]) -> Optional[str]:
    if f35mm is None:
        return None
    for known_fl, label in IPHONE_15_PRO_MAX_LENSES.items():
        if abs(f35mm - known_fl) <= FL_MATCH_TOLERANCE:
            return label
    return f"unrecognised ({f35mm:.0f} mm)"


# ── Intrinsic calculation ──────────────────────────────────────────────────────

def intrinsics_from_f35mm(
    f35mm: float,
    width: int,
    height: int,
    target_size: int = 256,
) -> Dict[str, Any]:
    """
    Convert 35mm-equiv focal length to pipeline pixel intrinsics.
    """
    hfov = 2.0 * math.atan(FRAME_35MM_W / (2.0 * f35mm))
    vfov = 2.0 * math.atan(FRAME_35MM_H / (2.0 * f35mm))

    fx_n = (width / 2.0) / math.tan(hfov / 2.0)
    fy_n = (height / 2.0) / math.tan(vfov / 2.0)
    cx_n = width / 2.0
    cy_n = height / 2.0

    side = min(width, height)
    cx_crop = cx_n - (width - side) / 2.0
    cy_crop = cy_n - (height - side) / 2.0
    scale = target_size / side

    pipeline = {
        "fx": round(fx_n * scale, 2),
        "fy": round(fy_n * scale, 2),
        "cx": round(cx_crop * scale, 2),
        "cy": round(cy_crop * scale, 2),
    }

    return {
        "native_resolution": {"width": width, "height": height},
        "native_intrinsics": {
            "fx": round(fx_n, 2),
            "fy": round(fy_n, 2),
            "cx": round(cx_n, 2),
            "cy": round(cy_n, 2),
        },
        "pipeline_intrinsics": pipeline,
        "computed_fov": {
            "hfov_deg": round(math.degrees(hfov), 2),
            "vfov_deg": round(math.degrees(vfov), 2),
        },
        "f35mm_used": f35mm,
        "target_size": target_size,
    }


def _check_lens_consistency(infos: List[Dict[str, Any]]) -> None:
    labels = [i["lens_label"] for i in infos if i["lens_label"] is not None]
    unique = set(labels)
    if len(unique) > 1:
        print(
            f"\n  WARNING: videos appear to be from different lenses: {unique}\n"
            "  Averaging intrinsics across different lenses will produce incorrect\n"
            "  parameters. Re-run with only videos from the same lens, or pass\n"
            "  separate --out files per lens group."
        )


def _average_pipeline_intrinsics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["fx", "fy", "cx", "cy"]
    avg: Dict[str, float] = {}
    for k in keys:
        vals = [r["pipeline_intrinsics"][k] for r in results]
        avg[k] = round(sum(vals) / len(vals), 2)
        spread = max(vals) - min(vals)
        if avg[k] > 0 and spread / avg[k] > 0.05:
            print(
                f"  WARNING: {k} varies by {spread:.1f} px across videos "
                f"({100 * spread / avg[k]:.1f}%) — check for mixed lens/camera setups"
            )
    return avg


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics from iPhone .mov metadata without ffprobe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
iPhone 15 Pro Max 35mm-equiv focal lengths (use with --f35mm_override if needed):
  0.5x ultrawide   :  13 mm
  1x  wide (main)  :  24 mm
  2x  tele crop    :  48 mm
  5x  periscope    : 120 mm
        """,
    )
    parser.add_argument(
        "--video", nargs="+", required=True,
        help="One or more .mov video files (must all be from the same lens)",
    )
    parser.add_argument(
        "--camera_height_m", type=float, required=True,
        help="Camera height above ground in metres",
    )
    parser.add_argument(
        "--f35mm_override", type=float, default=None,
        help="Override 35mm-equiv focal length (mm) when metadata extraction fails",
    )
    parser.add_argument(
        "--target_size", type=int, default=256,
        help="Pipeline working resolution in pixels (default: 256)",
    )
    parser.add_argument(
        "--out", type=str, default="camera_params.json",
        help="Output camera_params.json path",
    )
    args = parser.parse_args()

    video_paths = [Path(v) for v in args.video]
    for p in video_paths:
        if not p.exists():
            sys.exit(f"Video not found: {p}")

    print(f"Reading metadata from {len(video_paths)} video(s)...\n")

    infos: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    fps_set = set()

    for vp in video_paths:
        info = extract_video_info(vp)
        infos.append(info)
        fps_set.add(info["fps"])

        print(f"  {vp.name}")
        print(f"    Resolution : {info['width']} x {info['height']}")
        print(f"    FPS        : {info['fps']:.2f}")
        print(f"    f35mm      : {info['f35mm']}  (source: {info['fl_source']})")
        if info["lens_label"]:
            print(f"    Lens       : {info['lens_label']}")

        if args.f35mm_override is not None:
            f35mm = args.f35mm_override
            print(f"    Using --f35mm_override: {f35mm} mm")
        elif info["f35mm"] is not None:
            f35mm = info["f35mm"]
        else:
            f35mm = IPHONE_FALLBACK_FL["iphone_15_pro"]
            print(
                f"    WARNING: could not read focal length from metadata.\n"
                f"    Falling back to {f35mm} mm (iPhone 15 Pro 1x wide camera).\n"
                f"    Use --f35mm_override if this is wrong for your lens."
            )

        r = intrinsics_from_f35mm(f35mm, info["width"], info["height"], args.target_size)
        r["fps"] = info["fps"]
        r["fl_source"] = info["fl_source"]
        r["lens_label"] = info["lens_label"]
        results.append(r)

        p_intr = r["pipeline_intrinsics"]
        print(
            f"    Pipeline intrinsics ({args.target_size}px): "
            f"fx={p_intr['fx']:.1f}  fy={p_intr['fy']:.1f}  "
            f"cx={p_intr['cx']:.1f}  cy={p_intr['cy']:.1f}"
        )
        print(
            f"    FOV: {r['computed_fov']['hfov_deg']:.1f}° H  "
            f"{r['computed_fov']['vfov_deg']:.1f}° V"
        )
        print()

    if len(fps_set) > 1:
        print(
            f"  NOTE: mixed frame rates detected: {sorted(fps_set)}\n"
            "  The pipeline config should match the video being tested.\n"
        )
    else:
        print(f"  All videos are {next(iter(fps_set)):.0f} fps.\n")

    _check_lens_consistency(infos)

    if len(results) > 1:
        print(f"Averaging pipeline intrinsics across {len(results)} videos...")
        pipeline_avg = _average_pipeline_intrinsics(results)
    else:
        pipeline_avg = results[0]["pipeline_intrinsics"].copy()

    pipeline_avg["camera_height_m"] = args.camera_height_m

    print(
        f"\ncamera_height_m = {args.camera_height_m} m\n"
        "  A ±0.2 m error in height causes ~1–2 m carry error at 150 m.\n"
        "  Keep separate camera_params files for meaningfully different tripod heights."
    )

    output = {
        "method": "metadata_estimation_no_ffprobe",
        "note": (
            "Intrinsics estimated from video metadata / exiftool focal length, "
            "not checkerboard calibration. Expect some error in fx/fy vs full calibration."
        ),
        "native_resolution": results[0]["native_resolution"],
        "native_intrinsics": results[0]["native_intrinsics"],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "pipeline_intrinsics": pipeline_avg,
        "target_size": args.target_size,
        "f35mm_used": results[0]["f35mm_used"],
        "computed_fov": results[0]["computed_fov"],
        "detected_fps": sorted(fps_set),
        "lens_label": results[0]["lens_label"],
        "per_video": [
            {
                "fps": r["fps"],
                "lens_label": r["lens_label"],
                "f35mm_used": r["f35mm_used"],
                "fl_source": r["fl_source"],
                "pipeline_intrinsics": r["pipeline_intrinsics"],
            }
            for r in results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("  pipeline_intrinsics:")
    for k, v in pipeline_avg.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()