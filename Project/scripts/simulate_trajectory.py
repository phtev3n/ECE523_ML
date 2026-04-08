"""Reconstruct per-frame 3D trajectories from Trackman shot-level data.

Uses a physics model with aerodynamic drag and Magnus lift, integrated via
RK4 at 1/60s steps.  Per-shot Cd/Cm are optimized so simulated carry distance
and apex height match Trackman's reported values.

Usage
-----
python scripts/simulate_trajectory.py \
    --shots trackman_shots.json \
    --camera_distance_m 5.0 \
    --fps 60.0 \
    --num_frames 24 \
    --out trajectories.json

Output: list of per-shot dicts with "xyz" (Nx3 list) in camera coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# ---- Golf ball physical constants ----
BALL_MASS = 0.04593        # kg
BALL_RADIUS = 0.02134      # m
BALL_AREA = math.pi * BALL_RADIUS ** 2  # m^2
GRAVITY = 9.81             # m/s^2
AIR_DENSITY = 1.225        # kg/m^3 (sea level, 15 C)


def _accelerations(state: np.ndarray, Cd: float, Cm: float, spin_vec: np.ndarray) -> np.ndarray:
    """Compute accelerations from gravity + drag + Magnus."""
    vx, vy, vz = state[3], state[4], state[5]
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-6:
        return np.array([0.0, -GRAVITY, 0.0])

    # Drag: opposes velocity
    drag_coeff = 0.5 * AIR_DENSITY * Cd * BALL_AREA / BALL_MASS
    ax_drag = -drag_coeff * speed * vx
    ay_drag = -drag_coeff * speed * vy
    az_drag = -drag_coeff * speed * vz

    # Magnus: F_m proportional to (spin x velocity)
    magnus_coeff = 0.5 * AIR_DENSITY * Cm * BALL_AREA / BALL_MASS
    # spin_vec cross velocity
    sx, sy, sz = spin_vec
    mx = sy * vz - sz * vy
    my = sz * vx - sx * vz
    mz = sx * vy - sy * vx
    ax_magnus = magnus_coeff * speed * mx
    ay_magnus = magnus_coeff * speed * my
    az_magnus = magnus_coeff * speed * mz

    return np.array([
        ax_drag + ax_magnus,
        -GRAVITY + ay_drag + ay_magnus,
        az_drag + az_magnus,
    ])


def rk4_step(state: np.ndarray, dt: float, Cd: float, Cm: float, spin_vec: np.ndarray) -> np.ndarray:
    """One RK4 integration step. state = [x, y, z, vx, vy, vz]."""
    def deriv(s):
        acc = _accelerations(s, Cd, Cm, spin_vec)
        return np.array([s[3], s[4], s[5], acc[0], acc[1], acc[2]])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_shot(
    ball_speed: float,
    launch_angle_deg: float,
    launch_direction_deg: float,
    spin_rate_rpm: float,
    spin_axis_deg: float,
    Cd: float,
    Cm: float,
    dt: float = 1 / 60.0,
    max_time: float = 10.0,
) -> np.ndarray:
    """Simulate a full golf shot from launch to landing.

    Coordinate system (matches golf_tracer convention):
      X = forward / carry direction
      Y = up
      Z = lateral (side)

    Returns (N, 3) array of xyz positions at each dt step.
    """
    la = math.radians(launch_angle_deg)
    ld = math.radians(launch_direction_deg)

    vx0 = ball_speed * math.cos(la) * math.cos(ld)
    vy0 = ball_speed * math.sin(la)
    vz0 = ball_speed * math.cos(la) * math.sin(ld)

    # Spin vector from spin rate and spin axis.
    # Spin axis: 0° = pure backspin (lift in Y), positive = tilted right.
    # Backspin generates upward Magnus force.
    omega = spin_rate_rpm * 2.0 * math.pi / 60.0  # rad/s
    sa = math.radians(spin_axis_deg)
    # spin vector in body frame: backspin around Z axis, tilted by spin_axis
    spin_vec = np.array([
        0.0,
        omega * math.cos(sa),   # component generating lift (backspin)
        -omega * math.sin(sa),  # component generating curve (sidespin)
    ])

    state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])
    points = [state[:3].copy()]
    n_steps = int(max_time / dt)

    for _ in range(n_steps):
        state = rk4_step(state, dt, Cd, Cm, spin_vec)
        points.append(state[:3].copy())
        if state[1] < 0 and len(points) > 2:
            # Ball hit the ground — interpolate to y=0
            prev = points[-2]
            curr = state[:3]
            if curr[1] != prev[1]:
                frac = prev[1] / (prev[1] - curr[1])
                landing = prev + frac * (curr - prev)
                landing[1] = 0.0
                points[-1] = landing
            break

    return np.array(points)


def carry_distance(traj: np.ndarray) -> float:
    """Horizontal carry distance (XZ plane)."""
    dx = traj[-1, 0] - traj[0, 0]
    dz = traj[-1, 2] - traj[0, 2]
    return math.sqrt(dx * dx + dz * dz)


def apex_height(traj: np.ndarray) -> float:
    return float(np.max(traj[:, 1]))


def optimize_cd_cm(
    ball_speed: float,
    launch_angle_deg: float,
    launch_direction_deg: float,
    spin_rate_rpm: float,
    spin_axis_deg: float,
    target_carry_m: float,
    target_apex_m: float,
    dt: float = 1 / 60.0,
) -> tuple[float, float]:
    """Find Cd, Cm that match Trackman carry and apex for one shot."""

    def objective(params):
        Cd, Cm = params
        if Cd < 0.05 or Cd > 0.8 or Cm < 0.0 or Cm > 0.6:
            return 1e6
        traj = simulate_shot(ball_speed, launch_angle_deg, launch_direction_deg,
                             spin_rate_rpm, spin_axis_deg, Cd, Cm, dt)
        carry_err = (carry_distance(traj) - target_carry_m) ** 2
        apex_err = (apex_height(traj) - target_apex_m) ** 2
        return carry_err + apex_err

    result = minimize(objective, x0=[0.25, 0.18], method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 200})
    Cd_opt, Cm_opt = result.x
    Cd_opt = max(0.05, min(0.8, Cd_opt))
    Cm_opt = max(0.0, min(0.6, Cm_opt))
    return Cd_opt, Cm_opt


def to_camera_coords(traj: np.ndarray, camera_distance_m: float) -> np.ndarray:
    """Transform world trajectory to camera coordinates.

    The camera is *camera_distance_m* behind the ball along the Z-axis.
    In the pipeline convention, Z is depth from camera, so we offset Z
    to match the synthetic data convention (ball starts at Z~10).
    """
    out = traj.copy()
    out[:, 2] = traj[:, 2] + camera_distance_m  # Z offset so ball is in front of camera
    # Y stays the same (height above ground)
    # X stays the same (lateral/forward in world = X in camera)
    return out


def process_shots(
    shots: list[dict],
    camera_distance_m: float,
    fps: float,
    num_frames: int,
) -> list[dict]:
    """Process all shots: optimize, simulate, trim, transform."""
    dt = 1.0 / fps
    results = []

    for i, s in enumerate(shots):
        bs = s.get("ball_speed")
        la = s.get("launch_angle_deg")
        ld = s.get("launch_direction_deg", 0.0)
        sr = s.get("spin_rate_rpm", 2500.0)
        sa = s.get("spin_axis_deg", 0.0)
        carry = s.get("carry_m")
        apex = s.get("apex_m")

        if bs is None or la is None:
            print(f"  shot {i}: missing ball_speed or launch_angle, skipping")
            continue

        # Optimize Cd/Cm if we have Trackman endpoints
        if carry is not None and apex is not None and carry > 1 and apex > 0.5:
            Cd, Cm = optimize_cd_cm(bs, la, ld, sr, sa, carry, apex, dt)
        else:
            Cd, Cm = 0.25, 0.18

        traj = simulate_shot(bs, la, ld, sr, sa, Cd, Cm, dt)

        # Trim to num_frames (from launch)
        traj_trimmed = traj[:num_frames]
        if len(traj_trimmed) < num_frames:
            # Pad with last position if trajectory is shorter
            pad = np.tile(traj_trimmed[-1:], (num_frames - len(traj_trimmed), 1))
            traj_trimmed = np.concatenate([traj_trimmed, pad], axis=0)

        traj_camera = to_camera_coords(traj_trimmed, camera_distance_m)

        # Validate
        sim_carry = carry_distance(traj)
        sim_apex = apex_height(traj)

        result = {
            "shot_index": i,
            "trackman": s,
            "Cd": float(Cd),
            "Cm": float(Cm),
            "simulated_carry_m": float(sim_carry),
            "simulated_apex_m": float(sim_apex),
            "xyz": traj_camera.tolist(),
        }
        if carry is not None:
            result["carry_error_m"] = abs(sim_carry - carry)
        if apex is not None:
            result["apex_error_m"] = abs(sim_apex - apex)

        status = f"shot {i}: Cd={Cd:.3f} Cm={Cm:.3f} carry={sim_carry:.1f}m apex={sim_apex:.1f}m"
        if carry is not None:
            status += f" (target carry={carry:.1f}m err={result.get('carry_error_m', 0):.2f}m)"
        print(f"  {status}")
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate 3D trajectories from Trackman data")
    parser.add_argument("--shots", type=str, required=True, help="Path to trackman_shots.json")
    parser.add_argument("--camera_distance_m", type=float, default=5.0, help="Distance from camera to ball at address (meters)")
    parser.add_argument("--fps", type=float, default=60.0, help="Video frame rate")
    parser.add_argument("--num_frames", type=int, default=24, help="Frames per sequence")
    parser.add_argument("--out", type=str, default="trajectories.json", help="Output JSON path")
    args = parser.parse_args()

    shots_path = Path(args.shots)
    if not shots_path.exists():
        sys.exit(f"Shots file not found: {shots_path}")

    with open(shots_path) as f:
        shots = json.load(f)
    print(f"Loaded {len(shots)} shots")

    results = process_shots(shots, args.camera_distance_m, args.fps, args.num_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} trajectories to {out_path}")

    # Summary
    carry_errs = [r["carry_error_m"] for r in results if "carry_error_m" in r]
    apex_errs = [r["apex_error_m"] for r in results if "apex_error_m" in r]
    if carry_errs:
        print(f"Carry error: mean={np.mean(carry_errs):.2f}m  max={np.max(carry_errs):.2f}m")
    if apex_errs:
        print(f"Apex error:  mean={np.mean(apex_errs):.2f}m  max={np.max(apex_errs):.2f}m")


if __name__ == "__main__":
    main()
