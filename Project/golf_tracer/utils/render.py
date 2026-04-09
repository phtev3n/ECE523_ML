from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def draw_tracer(frame: np.ndarray, points_uv, color=(0, 255, 255), thickness: int = 2):
    """Draw a polyline tracer connecting sequential ball positions on a frame.

    Only finite points are included; NaN or Inf coordinates (from coasted
    Kalman predictions with no valid detection) are silently skipped.

    Args:
        frame      : BGR uint8 image to draw onto (modified in-place).
        points_uv  : Iterable of (u, v) pixel coordinates (the tracer history
                     up to the current frame).
        color      : BGR line colour.
        thickness  : Line thickness in pixels.
    """
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
    """Render a Trackman-style HUD metrics panel onto a video frame.

    The panel is drawn as a semi-transparent dark rectangle with a cyan accent
    border, a title row, a dividing line, and one labelled row per metric.
    Transparency is achieved via cv2.addWeighted on a copy of the frame, so the
    original frame is modified in-place and also returned.

    Metrics are sourced from compute_ball_metrics() (launch angle, carry, etc.)
    and optionally from the spin estimator (backspin / sidespin).  Any metric
    key absent from the dicts is silently omitted from the panel.

    Args:
        frame   : BGR uint8 image (modified in-place).
        metrics : dict from compute_ball_metrics().
        spin    : dict with 'backspin_rpm' and 'sidespin_rpm', or None to
                  omit spin rows (e.g. when clip is too short for estimation).
        origin  : (x, y) pixel coordinate of the panel's top-left corner.
        alpha   : Background opacity (0 = fully transparent, 1 = solid black).
    """
    # Build ordered list of (label, value_string) rows to display
    rows: list[tuple[str, str]] = []

    if "ball_speed_ms" in metrics:
        rows.append(("BALL SPEED",   f"{metrics['ball_speed_ms']:.1f} m/s"))
    if "launch_angle_deg" in metrics:
        rows.append(("LAUNCH ANG",   f"{metrics['launch_angle_deg']:.1f}\u00b0"))
    if "launch_direction_deg" in metrics:
        d    = metrics["launch_direction_deg"]
        side = "R" if d >= 0 else "L"
        rows.append(("DIRECTION",    f"{abs(d):.1f}\u00b0 {side}"))
    if "carry_m" in metrics:
        rows.append(("CARRY",        f"{metrics['carry_m']:.1f} m"))
    if "apex_m" in metrics:
        rows.append(("MAX HEIGHT",   f"{metrics['apex_m']:.1f} m"))
    if "descent_angle_deg" in metrics:
        rows.append(("DESCENT ANG",  f"{metrics['descent_angle_deg']:.1f}\u00b0"))
    if "time_of_flight_s" in metrics:
        rows.append(("FLIGHT TIME",  f"{metrics['time_of_flight_s']:.2f} s"))
    if spin:
        back      = spin.get("backspin_rpm")
        side_spin = spin.get("sidespin_rpm")
        if back is not None:
            rows.append(("BACKSPIN",  f"{int(round(back))} rpm"))
        if side_spin is not None:
            # Positive sidespin = draw (right-to-left curve for right-handers)
            # Negative sidespin = fade (left-to-right curve)
            label = "DRAW" if side_spin >= 0 else "FADE"
            rows.append((f"SIDESPIN ({label})", f"{abs(int(round(side_spin)))} rpm"))

    if not rows:
        return frame

    # Panel layout constants
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fs_title = 0.38
    fs_row   = 0.40
    fw       = 1
    line_h   = 18     # pixels per data row
    pad      = 7      # inner padding
    title_h  = 16     # height of the title row
    label_w  = 120    # pixel width reserved for the label column
    value_w  = 90     # pixel width reserved for the value column
    panel_w  = label_w + value_w + pad * 3
    panel_h  = len(rows) * line_h + pad * 2 + title_h + 4

    ox, oy = origin

    # Semi-transparent dark background: draw on a copy then blend
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + panel_w, oy + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Thin cyan accent border
    cv2.rectangle(frame, (ox, oy), (ox + panel_w, oy + panel_h), (0, 180, 220), 1)

    # Title row
    cv2.putText(
        frame, "BALL METRICS",
        (ox + pad, oy + pad + 10),
        font, fs_title, (0, 200, 255), fw, cv2.LINE_AA,
    )

    # Horizontal divider under title
    div_y = oy + pad + title_h
    cv2.line(frame, (ox + pad, div_y), (ox + panel_w - pad, div_y), (60, 60, 60), 1)

    # Data rows: grey label on left, white value on right
    for i, (label, value) in enumerate(rows):
        row_y = div_y + 4 + i * line_h + line_h - 4
        cv2.putText(frame, label, (ox + pad, row_y),
                    font, fs_row, (170, 170, 170), fw, cv2.LINE_AA)
        cv2.putText(frame, value, (ox + pad + label_w, row_y),
                    font, fs_row, (255, 255, 255), fw, cv2.LINE_AA)

    return frame


def render_video(frames, save_path: str | Path, fps: float = 60.0):
    """Write a list of BGR uint8 frames to an mp4v-encoded video file.

    Args:
        frames    : Sequence of (H, W, 3) BGR uint8 arrays.
        save_path : Output file path (.mp4 extension expected).
        fps       : Playback frame rate.
    """
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
