from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def draw_tracer(frame: np.ndarray, points_uv, color=(0, 255, 255), thickness: int = 2):
    pts = [(int(p[0]), int(p[1])) for p in points_uv if np.isfinite(p[0]) and np.isfinite(p[1])]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)
    return frame


def render_video(frames, save_path: str | Path, fps: float = 60.0):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
