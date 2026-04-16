from __future__ import annotations
import math
import numpy as np


def project_points(points_xyz: np.ndarray, camera: dict) -> np.ndarray:
    """Project 3D camera-frame points to 2D image-plane coordinates.

    Uses the 180°-rotated pinhole model matching frames extracted with
    --orient 180.  Both u and v increase in the opposite direction from the
    standard pinhole:
        u_rot = (W-1-cx) - fx * x / z
        v_rot = (H-1-cy) + fy * (y - camera_height) / z

    In particular, v INCREASES as the ball rises (y increases), matching the
    coordinate convention in the real-data annotations.

    Args:
        points_xyz : (N, 3) array of [x, y, z] world positions in metres.
                     x = lateral, y = height above ground, z = depth.
        camera     : dict with keys fx, fy, cx, cy, camera_height_m, and
                     optionally image_h / image_w (default 512).

    Returns:
        (N, 2) array of [u, v] image-plane pixel coordinates.
    """
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    camera_height = float(camera.get("camera_height_m", 0.0))
    H = float(camera.get("image_h", 512))
    W = float(camera.get("image_w", 512))
    z = np.clip(points_xyz[:, 2], 1e-3, None)   # prevent division by zero
    u = (W - 1 - cx) - fx * points_xyz[:, 0] / z
    v = (H - 1 - cy) + fy * (points_xyz[:, 1] - camera_height) / z
    return np.stack([u, v], axis=1)


def estimate_carry_from_xyz(points_xyz: np.ndarray) -> float:
    """Estimate carry distance as the horizontal displacement from first to last point.

    Carry is the ground-projected distance travelled (x-z plane), not the
    3D arc length.  This matches the Trackman definition: carry is measured
    from the impact point to where the ball first touches the ground.

    Note: for short clips that don't reach landing, this underestimates the
    true carry.  Use compute_ball_metrics for an extrapolated estimate.
    """
    if len(points_xyz) == 0:
        return 0.0
    start = points_xyz[0]
    end   = points_xyz[-1]
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    return float(math.sqrt(dx * dx + dz * dz))


def apex_height(points_xyz: np.ndarray) -> float:
    """Return the maximum observed y-coordinate (height above ground) in the clip.

    This is the *observed* apex within the provided frames.  For short clips
    that end before the ball peaks, this will underestimate the true apex.
    Use compute_ball_metrics for an analytically extrapolated apex.
    """
    if len(points_xyz) == 0:
        return 0.0
    return float(np.max(points_xyz[:, 1]))


def compute_ball_metrics(xyz_pred: np.ndarray, fps: float) -> dict:
    """Derive Trackman-style ball flight metrics from a predicted 3D trajectory.

    All metrics that depend on the full flight (carry, apex, ToF, descent angle)
    are extrapolated analytically from the initial velocity vector.  This is
    critical because the standard sequence length of 24 frames covers only
    ~0.4 s of flight — far shorter than a typical golf shot (2–4 s) — so
    reading carry or apex from the observed points alone would drastically
    underestimate them.

    Physics model used for extrapolation
    -------------------------------------
    Standard ballistic (no drag, no spin lift):
        carry    = horizontal_speed * t_land
        t_land   solved from: y0 + vy0*t - 0.5*g*t^2 = 0
        apex     = y0 + vy0^2 / (2g)    (at t = vy0/g)
        descent  = atan2(-vy_land, horiz_speed)
                   where vy_land = -sqrt(vy0^2 + 2*g*y0)

    Backspin lift will raise the true apex above this ballistic estimate.
    Once spin estimation is active (clips ≥ 90 frames), a corrected apex
    can be computed by numerically integrating the Magnus trajectory.

    Initial velocity is estimated via finite differences over the first
    min(4, T-1) frames, which smooths out single-frame detector noise.

    Returns:
        dict with keys: launch_angle_deg, launch_direction_deg, ball_speed_ms,
        carry_m, apex_m, descent_angle_deg, time_of_flight_s.
        Empty dict if fewer than 2 points are provided.
    """
    if len(xyz_pred) < 2:
        return {}

    dt = 1.0 / fps
    G  = 9.81

    # Estimate initial velocity over first min(4, T-1) frames
    n0 = min(4, len(xyz_pred) - 1)
    v0 = (xyz_pred[n0] - xyz_pred[0]) / (n0 * dt)
    vx0, vy0, vz0 = float(v0[0]), float(v0[1]), float(v0[2])

    speed   = math.sqrt(vx0 ** 2 + vy0 ** 2 + vz0 ** 2)

    # Physical plausibility guard: fastest recorded golf ball speed is ~91 m/s
    # (203 mph, long-drive competition).  LSTM predictions on out-of-distribution
    # shots can yield unrealistic velocities (e.g. 150 m/s), which causes carry
    # extrapolation to produce physically impossible distances (~500 m+).
    # Scale the full velocity vector proportionally so the direction is preserved.
    MAX_BALL_SPEED_MS = 91.0
    if speed > MAX_BALL_SPEED_MS:
        scale = MAX_BALL_SPEED_MS / speed
        vx0  *= scale
        vy0  *= scale
        vz0  *= scale
        speed = MAX_BALL_SPEED_MS

    horiz0  = math.sqrt(vx0 ** 2 + vz0 ** 2)   # horizontal component of speed

    # Launch angle: elevation above horizontal
    launch_angle = math.degrees(math.atan2(vy0, max(horiz0, 1e-6)))

    # Launch direction: lateral deviation from straight (positive = right)
    launch_direction = math.degrees(math.atan2(vx0, max(vz0, 1e-6)))

    # Time of flight: solve y0 + vy0*t - 0.5*g*t^2 = 0 for the positive root
    y0   = float(xyz_pred[0, 1])
    disc = vy0 ** 2 + 2.0 * G * y0
    if disc >= 0 and (vy0 >= 0 or disc > 0):
        t_land = (vy0 + math.sqrt(max(disc, 0.0))) / G
    else:
        t_land = (len(xyz_pred) - 1) * dt
    # Floor at observed window length so the estimate is never shorter than the clip
    t_land = max(t_land, (len(xyz_pred) - 1) * dt)

    # Carry: horizontal speed × time to land (assumes constant horiz speed, no drag)
    carry = horiz0 * t_land

    # Apex: peak height reached when vy = 0, i.e. t_apex = vy0 / g
    # Using initial conditions avoids the "clip ends before apex" problem where
    # max(xyz_pred[:,1]) would only report the height at the last observed frame.
    t_apex = vy0 / G if vy0 > 0 else 0.0
    apex   = y0 + vy0 * t_apex - 0.5 * G * t_apex ** 2

    # Descent angle: angle of the velocity vector at landing (energy conservation)
    # vy_land = -sqrt(vy0^2 + 2*g*y0); negative = downward.
    # This is correct even when the clip ends before the ball starts descending.
    vy_land_sq  = vy0 ** 2 + 2.0 * G * y0
    vy_land     = -math.sqrt(max(vy_land_sq, 0.0))
    descent_angle = math.degrees(math.atan2(-vy_land, max(horiz0, 1e-6)))

    return {
        "launch_angle_deg":     round(launch_angle, 1),
        "launch_direction_deg": round(launch_direction, 1),
        "ball_speed_ms":        round(speed, 1),
        "carry_m":              round(carry, 1),
        "apex_m":               round(apex, 2),
        "descent_angle_deg":    round(descent_angle, 1),
        "time_of_flight_s":     round(t_land, 2),
    }


def estimate_spin_from_trajectory(xyz_pred: np.ndarray, fps: float) -> dict | None:
    """Estimate backspin and sidespin by fitting a Magnus-force ballistic model
    to the observed 3D trajectory via least-squares optimisation.

    Method
    ------
    The same Magnus-force numerical integrator used in simulate_ballistics is
    run forward from the estimated launch conditions, and scipy L-BFGS-B
    minimises the mean squared positional error between the simulated and
    observed trajectories.

    Multiple random restarts are used because the objective landscape is
    near-flat for short clips (see below) and a single fixed starting point
    reliably returns that seed as the answer without converging.

    Minimum clip length
    -------------------
    Magnus curvature over the observation window must exceed the measurement
    noise for spin to be detectable.  The cumulative deflection is:
        Δ ≈ K_MAGNUS * ω * v * t² / 2
          ≈ 1.5e-4 * 300 rad/s * 60 m/s * t² / 2

    At t = 0.4 s (24 frames, 60 fps) this is ~0.05 m — well below the
    typical rmse_3d of ~1 m.  Returns None for clips shorter than 1.5 s
    (90 frames at 60 fps) to avoid reporting meaningless estimates.

    Returns:
        dict with keys backspin_rpm and sidespin_rpm, or None if the clip
        is too short for reliable spin estimation.

    Requires scipy: ``pip install scipy``
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("scipy is required for spin estimation: pip install scipy") from exc

    # Enforce minimum clip duration before attempting optimisation
    min_frames = int(round(1.5 * fps))
    if len(xyz_pred) < max(6, min_frames):
        return None

    K_MAGNUS = 1.5e-4   # m⁻¹; matches the value in simulate_ballistics
    G  = 9.81
    dt = 1.0 / fps
    T  = len(xyz_pred)

    # Estimate launch state from first few frames
    n0 = min(4, T - 1)
    v0 = (xyz_pred[n0] - xyz_pred[0]) / (n0 * dt)
    p0 = xyz_pred[0].copy()

    def _simulate(backspin_rpm: float, sidespin_rpm: float) -> np.ndarray:
        """Forward-integrate the Magnus trajectory for given spin values."""
        omega_x = -backspin_rpm * 2.0 * math.pi / 60.0   # negative = backspin
        omega_y =  sidespin_rpm * 2.0 * math.pi / 60.0
        x, y, z  = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        pts = [[x, y, z]]
        for _ in range(T - 1):
            ax = K_MAGNUS * (omega_y * vz)
            ay = -G + K_MAGNUS * (-omega_x * vz)
            az = K_MAGNUS * (omega_x * vy - omega_y * vx)
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x  += vx * dt
            y   = max(0.0, y + vy * dt)
            z  += vz * dt
            pts.append([x, y, z])
        return np.array(pts, dtype=np.float32)

    def _objective(params: np.ndarray) -> float:
        sim = _simulate(float(params[0]), float(params[1]))
        return float(np.mean((sim - xyz_pred) ** 2))

    bounds = [(0.0, 8000.0), (-3000.0, 3000.0)]

    # Multiple restarts to avoid converging to the seed when the landscape is flat.
    # Fixed candidates cover the physically plausible range; random draws add
    # diversity to catch unexpected minima.
    rng = np.random.default_rng(seed=42)
    candidates = [
        np.array([2500.0,    0.0]),    # nominal mid-iron prior
        np.array([1500.0,  500.0]),
        np.array([4000.0, -500.0]),
        np.array([3000.0, 1000.0]),
        np.array([1500.0, -800.0]),
    ]
    for _ in range(5):
        candidates.append(np.array([
            rng.uniform(500.0, 7000.0),
            rng.uniform(-2500.0, 2500.0),
        ]))

    best_result = None
    best_fun    = float("inf")
    for x0 in candidates:
        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-6},
        )
        if res.fun < best_fun:
            best_fun    = res.fun
            best_result = res

    backspin = float(np.clip(best_result.x[0],  0.0, 8000.0))
    sidespin = float(np.clip(best_result.x[1], -3000.0, 3000.0))
    return {
        "backspin_rpm": round(backspin),
        "sidespin_rpm": round(sidespin),
    }
