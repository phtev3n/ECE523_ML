"""Reconstruct approximate 3D trajectories from 2D annotations + camera calibration.

No launch monitor required. Uses the pinhole camera model and ballistic
physics constraints to fit a 3D trajectory that reprojects onto the
observed 2D ball positions.

The key insight: a golf ball follows a ballistic arc (parabola + drag).
Given calibrated camera intrinsics and known camera height, each 2D
observation constrains a ray in 3D.  Fitting a physically plausible
ballistic trajectory to those rays recovers the 3D path.

Usage
-----
python scripts/reconstruct_3d_from_2d.py \
    --annotations_2d annotations_2d.json \
    --camera_params camera_params.json \
    --camera_distance_m 5.0 \
    --fps 60.0 \
    --out trajectories.json

Output format matches simulate_trajectory.py so build_dataset.py works
with either source of 3D data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ---- Golf ball constants (same as simulate_trajectory.py) ----
BALL_MASS = 0.04593
BALL_RADIUS = 0.02134
BALL_AREA = math.pi * BALL_RADIUS ** 2
GRAVITY = 9.81
AIR_DENSITY = 1.225


# ---- Ballistic simulation (simplified: drag only, no Magnus) ----

def simulate_trajectory(
    speed: float,
    launch_angle_deg: float,
    launch_direction_deg: float,
    Cd: float,
    num_frames: int,
    dt: float,
    camera_distance_m: float,
) -> np.ndarray:
    """Forward-simulate a ballistic trajectory and return xyz in camera coords.

    Coordinate system (matches golf_tracer convention):
      X = lateral (left/right of target line)
      Y = up
      Z = depth from camera
    """
    la = math.radians(launch_angle_deg)
    ld = math.radians(launch_direction_deg)

    # Initial velocity in world frame (forward = +Z_world, up = +Y, lateral = +X)
    v_forward = speed * math.cos(la)
    vy = speed * math.sin(la)
    vx = v_forward * math.sin(ld)   # lateral
    vz = v_forward * math.cos(ld)   # forward (becomes depth in camera coords)

    drag_k = 0.5 * AIR_DENSITY * Cd * BALL_AREA / BALL_MASS
    state = np.array([0.0, 0.0, 0.0, vx, vy, vz], dtype=np.float64)
    points = []

    for _ in range(num_frames):
        x, y, z, svx, svy, svz = state
        # Camera coords: offset Z by camera distance
        points.append([x, max(0.0, y), z + camera_distance_m])

        spd = math.sqrt(svx**2 + svy**2 + svz**2)
        ax = -drag_k * spd * svx
        ay = -GRAVITY - drag_k * spd * svy
        az = -drag_k * spd * svz

        state[0] += svx * dt
        state[1] += svy * dt
        state[2] += svz * dt
        state[3] += ax * dt
        state[4] += ay * dt
        state[5] += az * dt

        if state[1] < 0:
            state[1] = 0.0
            state[4] = 0.0

    return np.array(points, dtype=np.float32)


def project_to_2d(xyz: np.ndarray, camera: dict) -> np.ndarray:
    """Pinhole projection matching golf_tracer convention."""
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    cam_h = float(camera.get("camera_height_m", 0.0))
    z = np.clip(xyz[:, 2], 1e-3, None)
    u = fx * xyz[:, 0] / z + cx
    v = fy * (cam_h - xyz[:, 1]) / z + cy
    return np.stack([u, v], axis=1)


def fit_trajectory_to_2d(
    uv_observed: np.ndarray,
    visible: np.ndarray,
    camera: dict,
    num_frames: int,
    fps: float,
    camera_distance_m: float,
) -> dict:
    """Fit ballistic parameters so the reprojected 3D arc matches 2D observations.

    Optimizes: [speed, launch_angle, launch_direction, Cd]
    Minimizes: sum of squared 2D reprojection errors on visible frames.
    """
    dt = 1.0 / fps
    mask = visible > 0.5

    if np.sum(mask) < 3:
        # Not enough visible frames — return a default straight-line estimate
        return _fallback_estimate(uv_observed, camera, num_frames, dt, camera_distance_m)

    def objective(params):
        speed, la_deg, ld_deg, Cd = params
        if speed < 10 or speed > 90 or Cd < 0.05 or Cd > 0.8:
            return 1e8
        if la_deg < 2 or la_deg > 55:
            return 1e8

        xyz = simulate_trajectory(speed, la_deg, ld_deg, Cd, num_frames, dt, camera_distance_m)
        uv_proj = project_to_2d(xyz, camera)

        # Reprojection error on visible frames only
        diff = uv_observed[mask] - uv_proj[mask]
        return float(np.sum(diff ** 2))

    # Initial guess from rough 2D motion
    uv_vis = uv_observed[mask]
    du = uv_vis[-1, 0] - uv_vis[0, 0]  # lateral pixel motion
    dv = uv_vis[-1, 1] - uv_vis[0, 1]  # vertical pixel motion (positive = downward)

    # Rough heuristics for initial guess
    init_speed = 55.0   # mid-range ball speed
    init_launch = 18.0  # mid-range launch angle
    init_dir = np.clip(np.degrees(np.arctan2(du, 100)), -15, 15)  # rough lateral from pixel drift
    init_Cd = 0.25

    result = minimize(
        objective,
        x0=[init_speed, init_launch, init_dir, init_Cd],
        method="Nelder-Mead",
        options={"xatol": 0.1, "fatol": 1.0, "maxiter": 500, "adaptive": True},
    )

    speed, la_deg, ld_deg, Cd = result.x
    speed = max(10.0, min(90.0, speed))
    la_deg = max(2.0, min(55.0, la_deg))
    ld_deg = max(-20.0, min(20.0, ld_deg))
    Cd = max(0.05, min(0.8, Cd))

    xyz = simulate_trajectory(speed, la_deg, ld_deg, Cd, num_frames, dt, camera_distance_m)
    uv_reproj = project_to_2d(xyz, camera)
    diff = uv_observed[mask] - uv_reproj[mask]
    reproj_err = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))

    return {
        "speed_ms": float(speed),
        "launch_angle_deg": float(la_deg),
        "launch_direction_deg": float(ld_deg),
        "Cd": float(Cd),
        "Cm": 0.0,
        "reproj_error_px": reproj_err,
        "optimizer_success": bool(result.success),
        "xyz": xyz.tolist(),
    }


def _fallback_estimate(uv_observed, camera, num_frames, dt, camera_distance_m):
    """Fallback for sequences with fewer than 3 visible frames."""
    xyz = simulate_trajectory(50.0, 15.0, 0.0, 0.25, num_frames, dt, camera_distance_m)
    return {
        "speed_ms": 50.0,
        "launch_angle_deg": 15.0,
        "launch_direction_deg": 0.0,
        "Cd": 0.25,
        "Cm": 0.0,
        "reproj_error_px": -1.0,
        "optimizer_success": False,
        "xyz": xyz.tolist(),
    }


def process_annotation_files(
    annotation_paths: list[Path],
    camera: dict,
    fps: float,
    camera_distance_m: float,
) -> list[dict]:
    """Process multiple annotation files into 3D trajectory estimates."""
    dt = 1.0 / fps
    results = []

    for i, ann_path in enumerate(annotation_paths):
        with open(ann_path) as f:
            ann = json.load(f)

        num_frames = len(ann)
        uv = np.array([a["uv"] for a in ann], dtype=np.float32)
        vis = np.array([a["visible"] for a in ann], dtype=np.float32)
        n_visible = int(np.sum(vis > 0.5))

        print(f"  shot {i}: {num_frames} frames, {n_visible} visible — fitting...", end="")
        result = fit_trajectory_to_2d(uv, vis, camera, num_frames, fps, camera_distance_m)
        result["shot_index"] = i
        result["source"] = "video_only_reconstruction"

        print(f" speed={result['speed_ms']:.1f}m/s launch={result['launch_angle_deg']:.1f}° "
              f"reproj={result['reproj_error_px']:.2f}px {'OK' if result['optimizer_success'] else 'WARN'}")
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D trajectories from 2D annotations (no launch monitor needed)"
    )
    parser.add_argument("--annotations_2d", type=str, nargs="+", required=True,
                        help="Path(s) to annotation JSON files from annotate_ball_2d.py")
    parser.add_argument("--camera_params", type=str, required=True,
                        help="Path to camera_params.json from calibrate_camera.py")
    parser.add_argument("--camera_distance_m", type=float, default=5.0,
                        help="Distance from camera to ball at address (meters)")
    parser.add_argument("--fps", type=float, default=60.0, help="Video frame rate")
    parser.add_argument("--out", type=str, default="trajectories.json", help="Output JSON path")
    args = parser.parse_args()

    with open(args.camera_params) as f:
        cam_data = json.load(f)
    camera = cam_data["pipeline_intrinsics"]

    ann_paths = [Path(p) for p in args.annotations_2d]
    for p in ann_paths:
        if not p.exists():
            sys.exit(f"Annotation file not found: {p}")

    print(f"Reconstructing 3D from {len(ann_paths)} shots (video-only, no launch monitor)")
    print(f"Camera: fx={camera['fx']:.1f} fy={camera['fy']:.1f} cam_dist={args.camera_distance_m}m")

    results = process_annotation_files(ann_paths, camera, args.fps, args.camera_distance_m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    errors = [r["reproj_error_px"] for r in results if r["reproj_error_px"] > 0]
    if errors:
        print(f"\nReprojection error: mean={np.mean(errors):.2f}px  max={np.max(errors):.2f}px")
    n_ok = sum(1 for r in results if r["optimizer_success"])
    print(f"Optimization succeeded: {n_ok}/{len(results)} shots")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
