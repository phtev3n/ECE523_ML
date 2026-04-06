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

    # Descent angle from last min(4, T-1) frames
    n_end = min(4, len(xyz_pred) - 1)
    v_end = (xyz_pred[-1] - xyz_pred[-1 - n_end]) / (n_end * dt)
    vx_e, vy_e, vz_e = float(v_end[0]), float(v_end[1]), float(v_end[2])
    horiz_e = math.sqrt(vx_e ** 2 + vz_e ** 2)
    descent_angle = math.degrees(math.atan2(max(-vy_e, 0.0), max(horiz_e, 1e-6)))

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

    if len(xyz_pred) < 6:
        return {"backspin_rpm": 0.0, "sidespin_rpm": 0.0}

    K_MAGNUS = 1.5e-4
    G = 9.81
    dt = 1.0 / fps
    T = len(xyz_pred)

    # Estimate initial state from first few frames
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

    result = minimize(
        _objective,
        x0=np.array([2500.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(0.0, 8000.0), (-3000.0, 3000.0)],
        options={"maxiter": 300, "ftol": 1e-4},
    )

    backspin = float(np.clip(result.x[0], 0.0, 8000.0))
    sidespin = float(np.clip(result.x[1], -3000.0, 3000.0))
    return {
        "backspin_rpm": round(backspin),
        "sidespin_rpm": round(sidespin),
    }
