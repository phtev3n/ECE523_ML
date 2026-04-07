from __future__ import annotations
import math
import numpy as np


def project_points(points_xyz: np.ndarray, camera: dict) -> np.ndarray:
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    z = np.clip(points_xyz[:, 2], 1e-3, None)
    u = fx * points_xyz[:, 0] / z + cx
    v = fy * points_xyz[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def estimate_carry_from_xyz(points_xyz: np.ndarray) -> float:
    if len(points_xyz) == 0:
        return 0.0
    start = points_xyz[0]
    end = points_xyz[-1]
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    return float(math.sqrt(dx * dx + dz * dz))


def apex_height(points_xyz: np.ndarray) -> float:
    if len(points_xyz) == 0:
        return 0.0
    return float(np.max(points_xyz[:, 1]))


def compute_ball_metrics(xyz_pred: np.ndarray, fps: float) -> dict:
    """Derive Trackman-style ball flight metrics from a predicted 3D trajectory.

    Initial and final velocities are estimated via finite differences.
    Carry and time-of-flight are extrapolated to landing using a simple
    ballistic model so short (24-frame) clips still report full-flight numbers.

    Returns a dict with keys:
      launch_angle_deg, launch_direction_deg, ball_speed_ms,
      carry_m, apex_m, descent_angle_deg, time_of_flight_s
    """
    if len(xyz_pred) < 2:
        return {}

    dt = 1.0 / fps
    G = 9.81

    # Estimate initial velocity over first min(4, T-1) frames
    n0 = min(4, len(xyz_pred) - 1)
    v0 = (xyz_pred[n0] - xyz_pred[0]) / (n0 * dt)
    vx0, vy0, vz0 = float(v0[0]), float(v0[1]), float(v0[2])

    speed = math.sqrt(vx0 ** 2 + vy0 ** 2 + vz0 ** 2)
    horiz0 = math.sqrt(vx0 ** 2 + vz0 ** 2)

    launch_angle = math.degrees(math.atan2(vy0, max(horiz0, 1e-6)))
    launch_direction = math.degrees(math.atan2(vx0, max(vz0, 1e-6)))

    # Extrapolate to landing: solve y0 + vy0*t - 0.5*G*t^2 = 0
    y0 = float(xyz_pred[0, 1])
    disc = vy0 ** 2 + 2.0 * G * y0
    if disc >= 0 and (vy0 >= 0 or disc > 0):
        t_land = (vy0 + math.sqrt(max(disc, 0.0))) / G
    else:
        t_land = (len(xyz_pred) - 1) * dt
    t_land = max(t_land, (len(xyz_pred) - 1) * dt)  # at least the observed window

    carry = horiz0 * t_land

    # Descent angle: extrapolate to landing via energy conservation.
    # vy at landing = -sqrt(vy0² + 2·g·y0); horizontal speed assumed constant.
    # This is correct even when the clip ends before the ball starts descending.
    vy_land_sq = vy0 ** 2 + 2.0 * G * y0
    vy_land = -math.sqrt(max(vy_land_sq, 0.0))   # negative = downward
    descent_angle = math.degrees(math.atan2(-vy_land, max(horiz0, 1e-6)))

    return {
        "launch_angle_deg": round(launch_angle, 1),
        "launch_direction_deg": round(launch_direction, 1),
        "ball_speed_ms": round(speed, 1),
        "carry_m": round(carry, 1),
        "apex_m": round(apex_height(xyz_pred), 2),
        "descent_angle_deg": round(descent_angle, 1),
        "time_of_flight_s": round(t_land, 2),
    }


def estimate_spin_from_trajectory(xyz_pred: np.ndarray, fps: float) -> dict:
    """Estimate backspin and sidespin by fitting a Magnus-force ballistic model
    to the observed 3D trajectory via least-squares optimisation.

    Uses the same K_MAGNUS constant as the synthetic data generator so the
    optimizer can accurately recover generating-spin on simulated data.

    Requires scipy (``pip install scipy``).

    Returns a dict with keys: backspin_rpm (float), sidespin_rpm (float).
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("scipy is required for spin estimation: pip install scipy") from exc

    # Magnus curvature over the observed window must exceed the expected
    # measurement noise level to be detectable.  At 60 fps the cumulative
    # lateral/vertical deflection from spin is:
    #   Δ ≈ K_MAGNUS * ω * v * t² / 2  ≈ 1.5e-4 * 300 * 60 * t² / 2
    # For t = 24/60 ≈ 0.4 s this is only ~0.05 m — well below a typical
    # rmse_3d of ~1 m.  Require at least 1.5 s of flight (90 frames at 60 fps).
    min_frames = int(round(1.5 * fps))
    if len(xyz_pred) < max(6, min_frames):
        return None

    K_MAGNUS = 1.5e-4
    G = 9.81
    dt = 1.0 / fps
    T = len(xyz_pred)

    n0 = min(4, T - 1)
    v0 = (xyz_pred[n0] - xyz_pred[0]) / (n0 * dt)
    p0 = xyz_pred[0].copy()

    def _simulate(backspin_rpm: float, sidespin_rpm: float) -> np.ndarray:
        omega_x = -backspin_rpm * 2.0 * math.pi / 60.0
        omega_y = sidespin_rpm * 2.0 * math.pi / 60.0
        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        pts = [[x, y, z]]
        for _ in range(T - 1):
            ax = K_MAGNUS * (omega_y * vz)
            ay = -G + K_MAGNUS * (-omega_x * vz)
            az = K_MAGNUS * (omega_x * vy - omega_y * vx)
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x += vx * dt
            y = max(0.0, y + vy * dt)
            z += vz * dt
            pts.append([x, y, z])
        return np.array(pts, dtype=np.float32)

    def _objective(params: np.ndarray) -> float:
        sim = _simulate(float(params[0]), float(params[1]))
        return float(np.mean((sim - xyz_pred) ** 2))

    bounds = [(0.0, 8000.0), (-3000.0, 3000.0)]

    # Random restarts: sample diverse starting points and keep the best.
    # The objective is near-flat for short clips, so a single fixed x0 gets
    # stuck at the initial guess.  Multiple restarts explore the parameter
    # space and reduce the risk of reporting the seed as the answer.
    rng = np.random.default_rng(seed=42)
    candidates = [
        np.array([2500.0, 0.0]),       # nominal prior
        np.array([1500.0, 500.0]),
        np.array([4000.0, -500.0]),
        np.array([3000.0, 1000.0]),
        np.array([1500.0, -800.0]),
    ]
    # Add a few random draws
    for _ in range(5):
        candidates.append(
            np.array([
                rng.uniform(500.0, 7000.0),
                rng.uniform(-2500.0, 2500.0),
            ])
        )

    best_result = None
    best_fun = float("inf")
    for x0 in candidates:
        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-6},
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    backspin = float(np.clip(best_result.x[0], 0.0, 8000.0))
    sidespin = float(np.clip(best_result.x[1], -3000.0, 3000.0))
    return {
        "backspin_rpm": round(backspin),
        "sidespin_rpm": round(sidespin),
    }
