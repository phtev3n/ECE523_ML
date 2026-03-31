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
