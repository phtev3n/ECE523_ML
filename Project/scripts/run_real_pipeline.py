"""Orchestrate the full real-data pipeline end-to-end.

Steps
-----
  1  detect    Auto-detect impact frames in each video      (detect_impact_frame.py)
  2  camera    Estimate camera intrinsics from EXIF metadata (estimate_camera_params.py)
  3  extract   Extract 24 frames per shot starting at impact (extract_frames.py)
  4  annotate  Interactive 2D ball annotation, one shot at a time (annotate_ball_2d.py)
  5  recon     Reconstruct 3D trajectories from 2D annotations (reconstruct_3d_from_2d.py)
  6  build     Assemble final dataset directory (build_dataset.py)

Each step checks whether its output already exists and skips if so, making
the script safe to re-run after interruption.  Use --from_step to force
re-execution from a specific step onwards.

Directory layout produced in --work_dir
---------------------------------------
  impact_frames.json          step 1
  camera_params.json          step 2
  shots/
    <video_stem>/
      frames/  000000.png …   step 3
      sequence_meta.json
      annotations_2d.json     step 4  (manual)
  trajectories.json           step 5  (omitted with --mode_2d_only)
  shot_map.json               step 6 input
  dataset/                    step 6 output
    seq_0000/ …
    build_report.json

Usage
-----
# Full pipeline from scratch:
python scripts/run_real_pipeline.py \\
    --video_dir "C:/Users/brian/Documents/UofA_Docs/ECE_523_ML/Project/real_video_data" \\
    --work_dir  real_data_work \\
    --camera_height_m 1.4 \\
    --camera_distance_m 5.0

# Resume after annotation is done (re-run steps 5 and 6):
python scripts/run_real_pipeline.py ... --from_step recon

# Skip impact-frame GUI (auto-detect only — still shows GUI by default):
python scripts/run_real_pipeline.py ... --no_gui_detect

# Build 2D-only dataset (skip step 5, no 3D trajectories required):
python scripts/run_real_pipeline.py ... --mode_2d_only

# Override focal length when EXIF extraction fails (common for newer iPhones):
python scripts/run_real_pipeline.py ... --f35mm_override 24
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Globals ────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = sys.executable  # use the same Python interpreter

STEPS = ["detect", "camera", "extract", "annotate", "recon", "build"]

# Video files to exclude (Trackman Studio has a different camera setup)
EXCLUDE_DIRS = {"trackman_studio"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command, exit with a clear message on failure."""
    print(f"\n  >> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n[PIPELINE] Step '{label}' failed (exit {result.returncode}). Fix the error above and re-run with --from_step {label}.")


def _find_videos(video_dir: Path) -> list[Path]:
    """Find all .MOV/.mov files excluding the Trackman_Studio subfolder.

    Uses resolved paths and a seen-set to deduplicate on Windows, where the
    filesystem is case-insensitive and rglob("*.MOV") also matches "*.mov",
    which would otherwise return every file twice.
    """
    seen: set[Path] = set()
    videos: list[Path] = []
    for p in sorted(video_dir.rglob("*")):
        if p.suffix.lower() != ".mov":
            continue
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if any(part.lower() in EXCLUDE_DIRS for part in p.parts):
            print(f"  [skip] {p.name}  (Trackman_Studio subfolder)")
            continue
        videos.append(p)
    return videos


def _load_json(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _modal_fps(shots: list[dict]) -> float:
    """Return the most common fps across all shots."""
    from collections import Counter
    counts = Counter(s["fps"] for s in shots)
    best = counts.most_common(1)[0][0]
    if len(counts) > 1:
        print(f"  WARNING: mixed fps detected {dict(counts)}. Using {best:.0f} fps for reconstruct/build.")
    return best


# ── Step implementations ───────────────────────────────────────────────────────

def step_detect(args: argparse.Namespace, videos: list[Path], work_dir: Path) -> list[dict]:
    """Step 1 — detect impact frames with optional GUI confirmation."""
    out = work_dir / "impact_frames.json"

    if out.exists():
        shots = _load_json(out)
        # Check if all videos are covered
        done = {s["video"] for s in shots}
        remaining = [v for v in videos if str(v) not in done and str(v.resolve()) not in done]
        if not remaining:
            print(f"  [detect] already complete ({len(shots)} shots)  →  {out}")
            return shots
        print(f"  [detect] resuming — {len(remaining)} video(s) still to process")
    else:
        remaining = videos

    cmd = [PY, str(SCRIPTS_DIR / "detect_impact_frame.py"),
           "--video"] + [str(v) for v in remaining] + [
           "--out", str(out),
           "--resume",
    ]
    if args.no_gui_detect:
        cmd.append("--no_gui")
    if args.orient >= 0:
        cmd += ["--orient", str(args.orient)]

    _run(cmd, "detect")
    return _load_json(out)


def step_camera(args: argparse.Namespace, videos: list[Path], work_dir: Path) -> dict:
    """Step 2 — estimate camera intrinsics from EXIF metadata."""
    out = work_dir / "camera_params.json"

    if out.exists():
        print(f"  [camera] already complete  →  {out}")
        return _load_json(out)

    cmd = [PY, str(SCRIPTS_DIR / "estimate_camera_params.py"),
           "--video"] + [str(v) for v in videos] + [
           "--camera_height_m", str(args.camera_height_m),
           "--out", str(out),
    ]
    if args.f35mm_override is not None:
        cmd += ["--f35mm_override", str(args.f35mm_override)]

    _run(cmd, "camera")
    return _load_json(out)


def step_extract(args: argparse.Namespace, shots: list[dict], work_dir: Path) -> list[Path]:
    """Step 3 — extract frames for each shot."""
    shots_dir = work_dir / "shots"
    shot_dirs: list[Path] = []

    for shot in shots:
        vpath = Path(shot["video"])
        shot_dir = shots_dir / vpath.stem
        frames_dir = shot_dir / "frames"
        shot_dirs.append(shot_dir)

        if frames_dir.exists() and any(frames_dir.glob("*.png")):
            n = len(list(frames_dir.glob("*.png")))
            print(f"  [extract] {vpath.name}  already done ({n} frames)")
            continue

        print(f"  [extract] {vpath.name}  impact_frame={shot['impact_frame']}")
        cmd = [
            PY, str(SCRIPTS_DIR / "extract_frames.py"),
            "--video",        str(vpath),
            "--impact_frame", str(shot["impact_frame"]),
            "--num_frames",   str(args.num_frames),
            "--out_dir",      str(frames_dir),
            "--size",         str(args.size),
        ]
        if args.orient >= 0:
            cmd += ["--orient", str(args.orient)]
        _run(cmd, "extract")

    return shot_dirs


def step_annotate(args: argparse.Namespace, shots: list[dict], shot_dirs: list[Path]) -> None:
    """Step 4 — interactive 2D ball annotation (one shot at a time)."""
    pending = []
    for shot, shot_dir in zip(shots, shot_dirs):
        ann = shot_dir / "annotations_2d.json"
        if ann.exists():
            print(f"  [annotate] {Path(shot['video']).name}  already annotated")
        else:
            pending.append((shot, shot_dir))

    if not pending:
        return

    total = len(pending)
    print(f"\n  [annotate] {total} shot(s) to annotate.  Controls:")
    print("    Left-click  — mark ball centre")
    print("    n           — ball not visible")
    print("    z           — undo last frame")
    print("    Enter/Space — confirm and advance")
    print("    q           — quit and save progress\n")

    for i, (shot, shot_dir) in enumerate(pending):
        name = Path(shot["video"]).name
        print(f"  [annotate] ({i+1}/{total}) {name}")
        _run([
            PY, str(SCRIPTS_DIR / "annotate_ball_2d.py"),
            "--frames_dir", str(shot_dir / "frames"),
            "--out",        str(shot_dir / "annotations_2d.json"),
            "--resume",
        ], "annotate")


def step_recon(
    args: argparse.Namespace,
    shots: list[dict],
    shot_dirs: list[Path],
    work_dir: Path,
) -> Path | None:
    """Step 5 — reconstruct 3D trajectories from 2D annotations."""
    if args.mode_2d_only:
        print("  [recon] skipped (--mode_2d_only)")
        return None

    out = work_dir / "trajectories.json"
    if out.exists():
        print(f"  [recon] already complete  →  {out}")
        return out

    # Collect annotation files — must exist (annotate step should have created them)
    ann_files: list[str] = []
    for shot, shot_dir in zip(shots, shot_dirs):
        ann = shot_dir / "annotations_2d.json"
        if not ann.exists():
            sys.exit(
                f"  [recon] Missing annotations for {Path(shot['video']).name}.\n"
                "  Complete the annotate step first, then re-run with --from_step recon."
            )
        ann_files.append(str(ann))

    fps = _modal_fps(shots)

    _run([
        PY, str(SCRIPTS_DIR / "reconstruct_3d_from_2d.py"),
        "--annotations_2d",     *ann_files,
        "--camera_params",      str(work_dir / "camera_params.json"),
        "--camera_distance_m",  str(args.camera_distance_m),
        "--fps",                str(fps),
        "--out",                str(out),
    ], "recon")

    return out


def step_build(
    args: argparse.Namespace,
    shots: list[dict],
    shot_dirs: list[Path],
    work_dir: Path,
    trajectories_path: Path | None,
) -> Path:
    """Step 6 — assemble final dataset directory."""
    dataset_dir = work_dir / "dataset"
    report = dataset_dir / "build_report.json"

    if report.exists():
        print(f"  [build] already complete  →  {dataset_dir}")
        return dataset_dir

    fps = _modal_fps(shots)

    # Build shot_map.json consumed by build_dataset.py
    shot_map: list[dict] = []
    for i, (shot, shot_dir) in enumerate(zip(shots, shot_dirs)):
        entry: dict = {
            "video":          shot["video"],
            "impact_frame":   shot["impact_frame"],
            "frames_dir":     str(shot_dir / "frames"),
            "annotations_2d": str(shot_dir / "annotations_2d.json"),
        }
        if not args.mode_2d_only:
            entry["trajectory_index"] = i
        shot_map.append(entry)

    shot_map_path = work_dir / "shot_map.json"
    _save_json(shot_map_path, shot_map)
    print(f"  [build] wrote {shot_map_path}")

    cmd = [
        PY, str(SCRIPTS_DIR / "build_dataset.py"),
        "--shot_map",      str(shot_map_path),
        "--camera_params", str(work_dir / "camera_params.json"),
        "--out_dir",       str(dataset_dir),
        "--fps",           str(fps),
    ]
    if args.mode_2d_only:
        cmd.append("--mode_2d_only")
    else:
        cmd += ["--trajectories", str(trajectories_path)]

    _run(cmd, "build")
    return dataset_dir


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end real golf shot pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video_dir", required=True,
        help="Directory containing .MOV files (Trackman_Studio subfolder auto-excluded)",
    )
    p.add_argument(
        "--work_dir", default="real_data_work",
        help="Root directory for all intermediate and final outputs (default: real_data_work)",
    )
    p.add_argument(
        "--camera_height_m", type=float, required=True,
        help="Camera height above ground in metres (measure at filming location)",
    )
    p.add_argument(
        "--camera_distance_m", type=float, default=5.0,
        help="Approximate distance from camera to ball at address in metres (default: 5.0)",
    )
    p.add_argument(
        "--num_frames", type=int, default=24,
        help="Frames to extract per shot starting at impact (default: 24)",
    )
    p.add_argument(
        "--size", type=int, default=512,
        help="Square frame size in pixels saved to disk (default: 512). "
             "Must match image_size in detector.yaml and pipeline.yaml.",
    )
    p.add_argument(
        "--f35mm_override", type=float, default=None,
        help="Override 35mm-equiv focal length when EXIF extraction fails (e.g. 24 for iPhone 1x wide)",
    )
    p.add_argument(
        "--from_step",
        choices=STEPS,
        default=None,
        help="Re-run from this step onwards, ignoring existing outputs for it and all later steps",
    )
    p.add_argument(
        "--no_gui_detect", action="store_true",
        help="Skip impact-frame confirmation GUI (auto-detect only)",
    )
    p.add_argument(
        "--orient", type=int, default=-1,
        help="Force frame orientation for the impact GUI: 0=none, 90=CW, 180=upside-down, 270=CCW. "
             "Use 180 if the video appears upside-down. Default: auto-detect from metadata.",
    )
    p.add_argument(
        "--mode_2d_only", action="store_true",
        help="Build 2D-only dataset — skip 3D reconstruction (step 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        sys.exit(f"--video_dir not found: {video_dir}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which steps should be forced to re-run
    force_from = STEPS.index(args.from_step) if args.from_step else len(STEPS)
    if args.from_step:
        print(f"[pipeline] Forcing re-run from step '{args.from_step}'")
        # Delete outputs for forced steps so the done-checks re-trigger
        forced_outputs = {
            "detect":   work_dir / "impact_frames.json",
            "camera":   work_dir / "camera_params.json",
            "recon":    work_dir / "trajectories.json",
            "build":    work_dir / "dataset" / "build_report.json",
        }
        for step in STEPS[force_from:]:
            p = forced_outputs.get(step)
            if p and p.exists():
                p.unlink()
                print(f"  deleted {p}")
            # For extract/annotate, per-shot outputs are NOT deleted automatically
            # (user may have done partial annotation — don't throw it away)
            if step in ("extract", "annotate"):
                print(f"  NOTE: step '{step}' — per-shot outputs in {work_dir/'shots'} are NOT deleted.")
                print(f"        Delete individual shots/<name>/frames or annotations_2d.json to re-run them.")

    print(f"\n[pipeline] Looking for videos in {video_dir}")
    videos = _find_videos(video_dir)
    if not videos:
        sys.exit(f"No .MOV files found under {video_dir} (excluding Trackman_Studio)")
    print(f"  Found {len(videos)} video(s):")
    for v in videos:
        print(f"    {v.name}")

    # ── Step 1: detect ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 1/6  detect — impact frame detection")
    print(f"{'='*64}")
    shots = step_detect(args, videos, work_dir)
    print(f"  {len(shots)} shot(s) with impact frames")

    # ── Step 2: camera ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 2/6  camera — intrinsic estimation")
    print(f"{'='*64}")
    step_camera(args, videos, work_dir)

    # ── Step 3: extract ───────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 3/6  extract — frame extraction ({args.num_frames} frames/shot)")
    print(f"{'='*64}")
    shot_dirs = step_extract(args, shots, work_dir)

    # ── Step 4: annotate ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 4/6  annotate — interactive 2D ball annotation")
    print(f"{'='*64}")
    step_annotate(args, shots, shot_dirs)

    # Check all annotations exist before proceeding
    missing_ann = [
        Path(shots[i]["video"]).name
        for i, sd in enumerate(shot_dirs)
        if not (sd / "annotations_2d.json").exists()
    ]
    if missing_ann:
        print(f"\n[pipeline] {len(missing_ann)} shot(s) still need annotation:")
        for n in missing_ann:
            print(f"    {n}")
        print(f"\nRe-run with --from_step annotate after completing annotation.")
        sys.exit(0)

    # ── Step 5: recon ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    mode_label = "SKIPPED (--mode_2d_only)" if args.mode_2d_only else "3D reconstruction"
    print(f"STEP 5/6  recon — {mode_label}")
    print(f"{'='*64}")
    trajectories_path = step_recon(args, shots, shot_dirs, work_dir)

    # ── Step 6: build ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"STEP 6/6  build — assemble dataset")
    print(f"{'='*64}")
    dataset_dir = step_build(args, shots, shot_dirs, work_dir, trajectories_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"[pipeline] COMPLETE")
    print(f"  Dataset : {dataset_dir}")
    print(f"  Shots   : {len(shots)}")
    print(f"  Mode    : {'2D-only' if args.mode_2d_only else '3D'}")
    print(f"\nNext steps:")
    print(f"  Fine-tune detector on real data:")
    print(f"    python scripts/train_detector.py --config configs/detector.yaml \\")
    print(f"        --dataset_root {dataset_dir}")
    print(f"  Run full pipeline test:")
    print(f"    python scripts/test_pipeline.py --config configs/pipeline.yaml \\")
    print(f"        --dataset_root {dataset_dir} \\")
    print(f"        --detector_ckpt outputs/detector_best.pt \\")
    print(f"        --trajectory_ckpt outputs/trajectory_best.pt --save_video")


if __name__ == "__main__":
    main()
