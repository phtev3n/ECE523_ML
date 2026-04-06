from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def draw_tracer(frame: np.ndarray, points_uv, color=(0, 255, 255), thickness: int = 2):
    pts = [(int(p[0]), int(p[1])) for p in points_uv if np.isfinite(p[0]) and np.isfinite(p[1])]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)
    return frame


def draw_metrics_panel(
    frame: np.ndarray,
    metrics: dict,
    spin: dict | None = None,
    origin: tuple[int, int] = (8, 8),
    alpha: float = 0.65,
) -> np.ndarray:
    """Render a Trackman-style HUD panel onto *frame* (BGR, modified in-place).

    Args:
        frame:   BGR uint8 image.
        metrics: dict from ``compute_ball_metrics`` (launch angle, carry, etc.).
        spin:    optional dict with ``backspin_rpm`` and ``sidespin_rpm``.
        origin:  pixel (x, y) of the top-left corner of the panel.
        alpha:   opacity of the dark background box (0 = invisible, 1 = solid).
    """
    rows: list[tuple[str, str]] = []

    if "ball_speed_ms" in metrics:
        rows.append(("BALL SPEED", f"{metrics['ball_speed_ms']:.1f} m/s"))
    if "launch_angle_deg" in metrics:
        rows.append(("LAUNCH ANG", f"{metrics['launch_angle_deg']:.1f}\u00b0"))
    if "launch_direction_deg" in metrics:
        d = metrics["launch_direction_deg"]
        side = "R" if d >= 0 else "L"
        rows.append(("DIRECTION", f"{abs(d):.1f}\u00b0 {side}"))
    if "carry_m" in metrics:
        rows.append(("CARRY", f"{metrics['carry_m']:.1f} m"))
    if "apex_m" in metrics:
        rows.append(("MAX HEIGHT", f"{metrics['apex_m']:.1f} m"))
    if "descent_angle_deg" in metrics:
        rows.append(("DESCENT ANG", f"{metrics['descent_angle_deg']:.1f}\u00b0"))
    if "time_of_flight_s" in metrics:
        rows.append(("FLIGHT TIME", f"{metrics['time_of_flight_s']:.2f} s"))
    if spin:
        back = spin.get("backspin_rpm")
        side_spin = spin.get("sidespin_rpm")
        if back is not None:
            rows.append(("BACKSPIN", f"{int(round(back))} rpm"))
        if side_spin is not None:
            label = "DRAW" if side_spin >= 0 else "FADE"
            rows.append((f"SIDESPIN ({label})", f"{abs(int(round(side_spin)))} rpm"))

    if not rows:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_title = 0.38
    fs_row = 0.40
    fw = 1
    line_h = 18
    pad = 7
    title_h = 16
    label_w = 120
    value_w = 90
    panel_w = label_w + value_w + pad * 3
    panel_h = len(rows) * line_h + pad * 2 + title_h + 4

    ox, oy = origin

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + panel_w, oy + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Thin accent border
    cv2.rectangle(frame, (ox, oy), (ox + panel_w, oy + panel_h), (0, 180, 220), 1)

    # Title row
    cv2.putText(
        frame, "BALL METRICS",
        (ox + pad, oy + pad + 10),
        font, fs_title, (0, 200, 255), fw, cv2.LINE_AA,
    )

    # Divider under title
    div_y = oy + pad + title_h
    cv2.line(frame, (ox + pad, div_y), (ox + panel_w - pad, div_y), (60, 60, 60), 1)

    # Data rows
    for i, (label, value) in enumerate(rows):
        row_y = div_y + 4 + i * line_h + line_h - 4
        cv2.putText(frame, label, (ox + pad, row_y),
                    font, fs_row, (170, 170, 170), fw, cv2.LINE_AA)
        cv2.putText(frame, value, (ox + pad + label_w, row_y),
                    font, fs_row, (255, 255, 255), fw, cv2.LINE_AA)

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
