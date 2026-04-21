"""Estimate camera intrinsics from iPhone .mov EXIF/QuickTime metadata.

When no checkerboard calibration is available, this script reads the focal
length and fps embedded in the video file and derives the pinhole camera
parameters needed by the golf_tracer pipeline.

Method
------
iPhone video files embed FocalLengthIn35mmFilm (or the physical FocalLength
plus sensor dimensions) in the QuickTime/MP4 metadata atoms.  From the
35mm-equivalent focal length the horizontal FOV is:

    HFOV = 2 * atan(36 / (2 * f_35mm))     [36 mm = full-frame sensor width]

From FOV and native video resolution:

    fx = (W / 2) / tan(HFOV / 2)
    fy = (H / 2) / tan(VFOV / 2)

cx = W/2, cy = H/2 is a good approximation for phone cameras.

The native intrinsics are then scaled to the pipeline's 256x256 working
resolution using the same centre-crop + resize transform as extract_frames.py.

iPhone 15 Pro Max lens map (35mm-equivalent focal lengths)
-----------------------------------------------------------
  0.5x  ultrawide          :  13 mm
  1x    wide (main camera) :  24 mm
  2x    telephoto crop     :  48 mm
  5x    periscope tele     : 120 mm

fps and lens are auto-detected from video metadata.  No --fps or zoom
flag is needed; the script reports what it found and warns on ambiguity.

Accuracy vs checkerboard
------------------------
~2–5% error on fx/fy → ~1–3 m carry error at 150 m.  Acceptable for
training data; re-calibrate with a checkerboard for sub-metre accuracy.

Requirements
------------
ffprobe (part of ffmpeg) must be on PATH.
exiftool (optional) provides richer metadata fallback on some devices.

Usage
-----
# Single video — auto-detects fps and lens:
python scripts/estimate_camera_params.py \\
    --video shots/IMG_0001.MOV \\
    --camera_height_m 1.4 \\
    --out camera_params.json

# Multiple videos — averages intrinsics (only mix same lens):
python scripts/estimate_camera_params.py \\
    --video shots/IMG_0001.MOV shots/IMG_0002.MOV \\
    --camera_height_m 1.4 \\
    --out camera_params.json

# Override focal length if EXIF extraction fails:
python scripts/estimate_camera_params.py \\
    --video shots/IMG_0001.MOV \\
    --camera_height_m 1.4 \\
    --f35mm_override 24 \\
    --out camera_params.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


# ── 35mm full-frame sensor reference dimensions ───────────────────────────────
FRAME_35MM_W = 36.0   # mm
FRAME_35MM_H = 24.0   # mm

# ── iPhone 15 Pro Max lens map: 35mm-equiv focal length → lens label ──────────
# Source: Apple iPhone 15 Pro Max tech specs.
# Used to identify which lens was active from the embedded focal length.
IPHONE_15_PRO_MAX_LENSES = {
    13:  "0.5x ultrawide",
    24:  "1x wide (main)",
    48:  "2x telephoto crop",
    120: "5x periscope tele",
}

# Tolerance (mm) for matching a reported focal length to a known lens
FL_MATCH_TOLERANCE = 2.0

# ── Per-model fallback table when EXIF extraction fails ──────────────────────
# Maps a rough model string to the 1x wide camera 35mm-equiv focal length.
IPHONE_FALLBACK_FL: dict[str, float] = {
    "iphone_15_pro":  24.0,
    "iphone_15":      26.0,
    "iphone_14_pro":  24.0,
    "iphone_14":      26.0,
    "iphone_13_pro":  26.0,
    "iphone_13":      26.0,
    "iphone_12_pro":  26.0,
    "iphone_12":      26.0,
    "default":        26.0,
}


# ── PyAV metadata helper (primary) ────────────────────────────────────────────

def _pyav_meta(video_path: Path) -> dict | None:
    """Return a ffprobe-compatible metadata dict using PyAV (no ffprobe needed).

    Builds the same nested structure that the rest of this file expects:
      {"streams": [{"codec_type": "video", "width": …, "tags": {…}}],
       "format":  {"tags": {…}}}
    Returns None if PyAV is not installed or the file cannot be opened.
    """
    try:
        import av  # optional — may not be installed in all environments
        with av.open(str(video_path)) as container:
            vstream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if vstream is None:
                return None

            # Dimensions: PyAV 9+ exposes width/height directly on the stream.
            # codec_context.width/height requires an open decoder and raises in
            # PyAV 17 when called on a demux-only container.
            width  = getattr(vstream, "width",  None)
            height = getattr(vstream, "height", None)
            if not width or not height:
                # Fallback: open codec context explicitly and read from there.
                with av.codec.CodecContext.create(vstream.codec.name, "r") as ctx:
                    ctx.extradata = vstream.codec_context.extradata
                    width  = width  or getattr(ctx, "width",  0)
                    height = height or getattr(ctx, "height", 0)
            if not width or not height:
                print(f"  [PyAV] could not read resolution from {video_path.name}",
                      file=sys.stderr)
                return None

            # fps — average_rate is a Fraction in PyAV; guard against plain floats.
            rate = vstream.average_rate or vstream.base_rate or vstream.guessed_rate
            fps_str = "60/1"
            if rate is not None:
                try:
                    fps_str = f"{rate.numerator}/{rate.denominator}"
                except AttributeError:
                    fps_str = f"{int(float(rate))}/1"

            # Metadata tags — both stream-level and container/format-level.
            stream_tags: dict = {}
            fmt_tags: dict = {}
            try:
                stream_tags = dict(vstream.metadata or {})
            except Exception:
                pass
            try:
                fmt_tags = dict(
                    getattr(getattr(container, "format", None), "metadata", None) or {}
                )
            except Exception:
                pass

            return {
                "streams": [{
                    "codec_type":   "video",
                    "width":        width,
                    "height":       height,
                    "r_frame_rate": fps_str,
                    "tags":         stream_tags,
                }],
                "format": {"tags": fmt_tags},
            }
    except Exception as exc:
        print(f"  [PyAV metadata failed for {video_path.name}]: {exc}", file=sys.stderr)
        return None


# ── ffprobe fallback (used only when PyAV unavailable / fails) ────────────────

def _run_ffprobe(video_path: Path) -> dict:
    """Return the full ffprobe JSON for *video_path*."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(r.stdout)
    except FileNotFoundError:
        sys.exit(
            "Neither PyAV nor ffprobe is available.\n"
            "Install PyAV:    pip install av\n"
            "  — or —\n"
            "Install ffmpeg:  winget install ffmpeg  (Windows)\n"
            "                 brew install ffmpeg    (macOS)"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffprobe failed on {video_path}: {e.stderr}")


def _try_exiftool_fl(video_path: Path) -> float | None:
    """Return FocalLengthIn35mmFilm via exiftool, or None."""
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

def extract_video_info(video_path: Path) -> dict:
    """Extract resolution, fps, and focal length from video metadata.

    Tries metadata sources in this order:
      1. ffprobe stream/format tags (covers most iPhone .mov files)
      2. exiftool FocalLengthIn35mmFilm (richer atom parsing)
      3. User-supplied --f35mm_override

    Returns a dict with keys:
        width, height   — native frame dimensions (px)
        fps             — detected frame rate (float)
        f35mm           — 35mm-equiv focal length (mm), or None
        fl_source       — description of where f35mm came from
        lens_label      — matched lens name (e.g. "1x wide (main)"), or None
    """
    # Try PyAV first (no external binary required); fall back to ffprobe.
    meta = _pyav_meta(video_path) or _run_ffprobe(video_path)

    # ── Find video stream ──
    video_stream = next(
        (s for s in meta.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        sys.exit(f"No video stream found in {video_path}")

    width  = int(video_stream["width"])
    height = int(video_stream["height"])

    # ── fps: prefer r_frame_rate (exact rational) ──
    fps = _parse_fps(
        video_stream.get("r_frame_rate")
        or video_stream.get("avg_frame_rate")
        or meta.get("format", {}).get("tags", {}).get("com.apple.quicktime.framerate")
        or "60/1"
    )

    # ── Focal length: search all tag dicts ──
    f35mm, fl_source = _extract_focal_length(video_stream, meta)
    if f35mm is None:
        exif_f = _try_exiftool_fl(video_path)
        if exif_f is not None:
            f35mm, fl_source = exif_f, "exiftool: FocalLengthIn35mmFilm"

    # ── Match to known lens ──
    lens_label = _match_lens(f35mm)

    return {
        "width":       width,
        "height":      height,
        "fps":         fps,
        "f35mm":       f35mm,
        "fl_source":   fl_source,
        "lens_label":  lens_label,
    }


def _parse_fps(fps_str: str) -> float:
    """Parse fps from a rational string like '60000/1001' or '30/1'."""
    try:
        parts = str(fps_str).split("/")
        if len(parts) == 2:
            val = float(parts[0]) / float(parts[1])
        else:
            val = float(parts[0])
        # Round common fractional rates to nearest integer for clean reporting
        # (59.94 → 60, 29.97 → 30)
        for standard in (24, 25, 30, 48, 50, 60, 120, 240):
            if abs(val - standard) < 0.5:
                return float(standard)
        return val
    except (ValueError, ZeroDivisionError):
        return 60.0


def _extract_focal_length(video_stream: dict, meta: dict) -> tuple[float | None, str]:
    """Search ffprobe tag dicts for a focal length value."""
    tag_dicts = [
        (video_stream.get("tags", {}),          "stream tag"),
        (meta.get("format", {}).get("tags", {}), "format tag"),
    ]
    focal_keys = [
        "com.apple.quicktime.camera.focal_length_35mm_equivalent",
        "com.apple.quicktime.focal_length_35mm",
        "focallengthin35mmfilm",
        "focal_length_35mm",
        "focal_length",
        "focallength",
    ]
    for tags, source_label in tag_dicts:
        # Normalise keys to lowercase for case-insensitive matching
        lower_tags = {k.lower(): v for k, v in tags.items()}
        for key in focal_keys:
            val = lower_tags.get(key)
            if val is not None:
                try:
                    f = float(str(val).replace("mm", "").strip())
                    if 10 < f < 200:
                        return f, f"{source_label}: {key}"
                except (ValueError, TypeError):
                    pass
        # Broad scan: any tag containing "focal"
        for key, val in lower_tags.items():
            if "focal" in key:
                try:
                    f = float(str(val).replace("mm", "").strip())
                    if 10 < f < 200:
                        return f, f"{source_label}: {key}"
                except (ValueError, TypeError):
                    pass
    return None, "not found"


def _match_lens(f35mm: float | None) -> str | None:
    """Match a focal length to a known iPhone 15 Pro Max lens label."""
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
) -> dict:
    """Convert 35mm-equiv focal length to pipeline pixel intrinsics.

    Scales to *target_size* using the same centre-crop-to-square + resize
    transform applied by extract_frames.py.
    """
    hfov = 2.0 * math.atan(FRAME_35MM_W / (2.0 * f35mm))
    vfov = 2.0 * math.atan(FRAME_35MM_H / (2.0 * f35mm))

    fx_n = (width  / 2.0) / math.tan(hfov / 2.0)
    fy_n = (height / 2.0) / math.tan(vfov / 2.0)
    cx_n = width  / 2.0
    cy_n = height / 2.0

    # Centre-crop to square then scale
    side    = min(width, height)
    cx_crop = cx_n - (width  - side) / 2.0
    cy_crop = cy_n - (height - side) / 2.0
    scale   = target_size / side

    pipeline = {
        "fx": round(fx_n * scale, 2),
        "fy": round(fy_n * scale, 2),
        "cx": round(cx_crop * scale, 2),
        "cy": round(cy_crop * scale, 2),
    }

    return {
        "native_resolution":  {"width": width, "height": height},
        "native_intrinsics":  {"fx": round(fx_n, 2), "fy": round(fy_n, 2),
                               "cx": round(cx_n, 2), "cy": round(cy_n, 2)},
        "pipeline_intrinsics": pipeline,
        "computed_fov": {
            "hfov_deg": round(math.degrees(hfov), 2),
            "vfov_deg": round(math.degrees(vfov), 2),
        },
        "f35mm_used": f35mm,
        "target_size": target_size,
    }


def _check_lens_consistency(infos: list[dict]) -> None:
    """Warn if videos were shot on different lenses."""
    labels = [i["lens_label"] for i in infos if i["lens_label"] is not None]
    unique = set(labels)
    if len(unique) > 1:
        print(
            f"\n  WARNING: videos appear to be from different lenses: {unique}\n"
            "  Averaging intrinsics across different lenses will produce incorrect\n"
            "  parameters.  Re-run with only videos from the same lens, or pass\n"
            "  separate --out files per lens group."
        )


def _average_pipeline_intrinsics(results: list[dict]) -> dict:
    """Average pipeline intrinsics across multiple same-lens videos."""
    keys = ["fx", "fy", "cx", "cy"]
    avg = {}
    for k in keys:
        vals = [r["pipeline_intrinsics"][k] for r in results]
        avg[k] = round(sum(vals) / len(vals), 2)
        spread = max(vals) - min(vals)
        if avg[k] > 0 and spread / avg[k] > 0.05:
            print(
                f"  WARNING: {k} varies by {spread:.1f} px across videos "
                f"({100*spread/avg[k]:.1f}%) — check for mixed lens/camera setups"
            )
    return avg


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics from iPhone .mov EXIF metadata",
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
        help="Camera height above ground in metres (measure at the filming location)",
    )
    parser.add_argument(
        "--f35mm_override", type=float, default=None,
        help="Override 35mm-equiv focal length (mm) when EXIF extraction fails",
    )
    parser.add_argument(
        "--target_size", type=int, default=256,
        help="Pipeline working resolution in pixels (default: 256)",
    )
    parser.add_argument(
        "--out", type=str, default="camera_params.json",
        help="Output camera_params.json path",
    )
    parser.add_argument(
        "--correct_existing", type=str, default=None,
        metavar="BAD_PARAMS_JSON",
        help=(
            "Read an existing camera_params.json produced by bad checkerboard "
            "calibration, recompute pipeline_intrinsics with centered cx/cy and "
            "EXIF-derived focal length, and write corrected output to --out.  "
            "Pass --f35mm_override if EXIF extraction fails."
        ),
    )
    args = parser.parse_args()

    # --- Mode: correct an existing bad calibration ---
    if args.correct_existing:
        bad_path = Path(args.correct_existing)
        if not bad_path.exists():
            sys.exit(f"Bad calibration file not found: {bad_path}")
        with open(bad_path) as fh:
            bad = json.load(fh)
        native = bad.get("native_resolution", {})
        width  = int(native.get("width",  4032))
        height = int(native.get("height", 3024))
        cam_h  = float(bad.get("pipeline_intrinsics", {}).get("camera_height_m", args.camera_height_m))

        if args.f35mm_override is not None:
            f35mm = args.f35mm_override
            fl_src = f"--f35mm_override: {f35mm} mm"
        else:
            # Try to read from a video file if supplied
            if args.video:
                info = extract_video_info(Path(args.video[0]))
                f35mm = info["f35mm"] or IPHONE_FALLBACK_FL["iphone_15_pro"]
                fl_src = info["fl_source"]
            else:
                f35mm = IPHONE_FALLBACK_FL["iphone_15_pro"]
                fl_src = "fallback (iPhone 15 Pro 1x wide)"
                print(f"  No video supplied; using fallback f35mm={f35mm} mm ({fl_src})")

        r = intrinsics_from_f35mm(f35mm, width, height, args.target_size)
        r["pipeline_intrinsics"]["camera_height_m"] = cam_h
        print(f"Correcting {bad_path.name}:")
        print(f"  f35mm={f35mm} mm  ({fl_src})")
        print(f"  Old pipeline cx={bad.get('pipeline_intrinsics', {}).get('cx', '?')}  cy={bad.get('pipeline_intrinsics', {}).get('cy', '?')}")
        print(f"  New pipeline cx={r['pipeline_intrinsics']['cx']}  cy={r['pipeline_intrinsics']['cy']}")
        output = {
            "method": "exif_estimation_corrected",
            "note": (
                "Intrinsics corrected from bad checkerboard calibration using EXIF focal length. "
                f"Prior file: {bad_path}. "
                "Re-run with --video to verify focal length from actual footage."
            ),
            "native_resolution":       {"width": width, "height": height},
            "native_intrinsics":       r["native_intrinsics"],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pipeline_intrinsics":     r["pipeline_intrinsics"],
            "target_size":             args.target_size,
            "f35mm_used":              f35mm,
            "computed_fov":            r["computed_fov"],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"Saved corrected intrinsics to {out_path}")
        return

    video_paths = [Path(v) for v in args.video]
    for p in video_paths:
        if not p.exists():
            sys.exit(f"Video not found: {p}")

    print(f"Reading metadata from {len(video_paths)} video(s)...\n")

    infos     = []
    results   = []
    fps_set   = set()

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

        # Determine focal length to use
        if args.f35mm_override is not None:
            f35mm = args.f35mm_override
            print(f"    Using --f35mm_override: {f35mm} mm")
        elif info["f35mm"] is not None:
            f35mm = info["f35mm"]
        else:
            f35mm = IPHONE_FALLBACK_FL["iphone_15_pro"]
            print(
                f"    WARNING: could not read focal length from EXIF.\n"
                f"    Falling back to {f35mm} mm (iPhone 15 Pro 1x wide camera).\n"
                f"    Use --f35mm_override if this is wrong for your lens."
            )

        r = intrinsics_from_f35mm(f35mm, info["width"], info["height"], args.target_size)
        r["fps"]        = info["fps"]
        r["fl_source"]  = info["fl_source"]
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

    # ── fps summary ──
    if len(fps_set) > 1:
        print(
            f"  NOTE: mixed frame rates detected: {sorted(fps_set)}\n"
            "  Both are supported.  extract_frames.py auto-detects fps per video,\n"
            "  so no manual --fps flag is needed.  The pipeline config (pipeline.yaml)\n"
            "  fps field should match the video being tested.\n"
        )
    else:
        print(f"  All videos are {next(iter(fps_set)):.0f} fps.\n")

    # ── lens consistency check ──
    _check_lens_consistency(infos)

    # ── Average intrinsics across videos ──
    if len(results) > 1:
        print(f"Averaging pipeline intrinsics across {len(results)} videos...")
        pipeline_avg = _average_pipeline_intrinsics(results)
    else:
        pipeline_avg = results[0]["pipeline_intrinsics"].copy()

    pipeline_avg["camera_height_m"] = args.camera_height_m

    # ── camera_height_m advice ──
    print(
        f"\ncamera_height_m = {args.camera_height_m} m\n"
        "  A ±0.2 m error in height causes ~1–2 m carry error at 150 m.\n"
        "  If you filmed some shots at a noticeably different tripod height,\n"
        "  run this script again with the other height and keep separate\n"
        "  camera_params files for each setup."
    )

    # ── Build output matching calibrate_camera.py format ──
    output = {
        "method": "exif_estimation",
        "note": (
            "Intrinsics estimated from EXIF focal length, not checkerboard calibration. "
            "Expect ~2-5% error in fx/fy vs a full calibration. "
            "Re-run calibrate_camera.py with checkerboard images for higher accuracy."
        ),
        "native_resolution":       results[0]["native_resolution"],
        "native_intrinsics":       results[0]["native_intrinsics"],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "pipeline_intrinsics":     pipeline_avg,
        "target_size":             args.target_size,
        "f35mm_used":              results[0]["f35mm_used"],
        "computed_fov":            results[0]["computed_fov"],
        "detected_fps":            sorted(fps_set),
        "lens_label":              results[0]["lens_label"],
        "per_video":               [
            {
                "fps":        r["fps"],
                "lens_label": r["lens_label"],
                "f35mm_used": r["f35mm_used"],
                "fl_source":  r["fl_source"],
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
