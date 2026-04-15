"""Animated 3D golf ball trajectory viewer.

Reads seq_XXXX_predictions.json files produced by test_pipeline.py and shows
a side-by-side animated simulation of the predicted ball flight:
  - Left panel  : Profile view  (downrange Z vs height Y)
  - Right panel : Top-down view (lateral X vs downrange Z)

The animation plays the captured frames at real speed then extrapolates
the full ballistic flight to landing using the initial velocity estimate,
so you see both the measured clip and the predicted complete trajectory.

Usage
-----
# View all sequences in the default test_results folder:
python scripts/view_trajectories.py

# View a specific folder or specific sequence:
python scripts/view_trajectories.py --results_dir outputs/test_results
python scripts/view_trajectories.py --seq 4          # only seq_0004

Controls
--------
  Space / Enter   : pause / resume animation
  n               : next sequence
  p               : previous sequence
  r               : restart current sequence
  q / Esc         : quit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "text.color":        "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#2a2a2a",
    "grid.linewidth":    0.6,
})

G = 9.81  # m/s²


# ── Physics helpers ────────────────────────────────────────────────────────────

def extrapolate_trajectory(xyz: np.ndarray, fps: float, n_extra: int = 300) -> np.ndarray:
    """Extend the observed xyz clip to landing using ballistic extrapolation.

    Estimates the initial velocity from the first few frames, then forward-
    integrates a drag-free ballistic arc until y ≤ 0.

    Returns the full trajectory (observed + extrapolated) as (M, 3) array.
    """
    if len(xyz) < 2:
        return xyz

    dt = 1.0 / fps
    n0 = min(4, len(xyz) - 1)
    v0 = (xyz[n0] - xyz[0]) / (n0 * dt)
    vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])

    pts = list(xyz)
    x, y, z = float(xyz[-1, 0]), float(xyz[-1, 1]), float(xyz[-1, 2])

    for _ in range(n_extra):
        x += vx * dt
        y += vy * dt
        z += vz * dt
        vy -= G * dt
        pts.append([x, y, z])
        if y <= 0.0:
            # Clamp landing point to ground
            pts[-1][1] = 0.0
            break

    return np.array(pts, dtype=np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sequence(json_path: Path) -> dict | None:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Could not load {json_path.name}: {e}")
        return None

    xyz = np.asarray(data.get("xyz_pred", []), dtype=np.float32)
    if len(xyz) == 0:
        return None

    metrics   = data.get("ball_metrics", {}) or {}
    spin      = data.get("spin")
    fps       = float(data.get("metrics", {}).get("ball_time_of_flight_s", 2.0))

    # Try to get fps from the metrics dict fps field; fall back to 60
    # We stored fps_seq from the dataset — not directly in predictions.json,
    # so we infer it from Kalman dt assumption (60fps for annotated shots).
    seq_fps   = 60.0

    full_xyz  = extrapolate_trajectory(xyz, seq_fps, n_extra=600)
    n_obs     = len(xyz)

    return {
        "name":       json_path.stem.replace("_predictions", ""),
        "xyz_obs":    xyz,
        "xyz_full":   full_xyz,
        "n_obs":      n_obs,
        "metrics":    metrics,
        "spin":       spin,
        "fps":        seq_fps,
    }


def find_sequences(results_dir: Path, only_seq: int | None) -> list[dict]:
    pattern = f"seq_{only_seq:04d}_predictions.json" if only_seq is not None else "*_predictions.json"
    paths = sorted(results_dir.glob(pattern))
    seqs = []
    for p in paths:
        s = load_sequence(p)
        if s is not None:
            seqs.append(s)
    return seqs


# ── Plotting helpers ───────────────────────────────────────────────────────────

_COL_OBS   = "#00d4ff"   # cyan  — observed (within clip)
_COL_EXT   = "#ff8c00"   # amber — extrapolated flight
_COL_BALL  = "#ffffff"   # white — current ball position dot
_COL_LAND  = "#ff4444"   # red   — landing marker
_COL_APEX  = "#44ff88"   # green — apex marker


def _format_metrics(metrics: dict, spin: dict | None) -> str:
    lines = []
    if "ball_speed_ms" in metrics:
        ms  = metrics["ball_speed_ms"]
        mph = ms * 2.23694
        lines.append(f"Speed   {ms:.1f} m/s  ({mph:.0f} mph)")
    if "launch_angle_deg" in metrics:
        lines.append(f"Launch  {metrics['launch_angle_deg']:.1f}°")
    if "carry_m" in metrics:
        m  = metrics["carry_m"]
        yd = m * 1.09361
        lines.append(f"Carry   {m:.0f} m  ({yd:.0f} yd)")
    if "apex_m" in metrics:
        m  = metrics["apex_m"]
        ft = m * 3.28084
        lines.append(f"Apex    {m:.1f} m  ({ft:.0f} ft)")
    if "descent_angle_deg" in metrics:
        lines.append(f"Descent {metrics['descent_angle_deg']:.1f}°")
    if "time_of_flight_s" in metrics:
        lines.append(f"ToF     {metrics['time_of_flight_s']:.2f} s")
    if spin and spin.get("backspin_rpm", 0) > 100:
        lines.append(f"Backsp  {spin['backspin_rpm']:.0f} rpm")
    return "\n".join(lines)


# ── Animator ──────────────────────────────────────────────────────────────────

class TrajectoryAnimator:
    """Drives a matplotlib figure with profile + top-down trajectory animation."""

    ANIM_FPS   = 30           # animation playback rate (independent of golf fps)
    TRAIL_LEN  = 40           # frames of trail to keep visible

    def __init__(self, sequences: list[dict]):
        self.sequences  = sequences
        self.seq_idx    = 0
        self.paused     = False
        self.frame_idx  = 0
        self._anim      = None

        self.fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Golf Trajectory Viewer")

        gs = self.fig.add_gridspec(1, 3, width_ratios=[5, 5, 2])
        self.ax_profile = self.fig.add_subplot(gs[0])
        self.ax_top     = self.fig.add_subplot(gs[1])
        self.ax_text    = self.fig.add_subplot(gs[2])
        self.ax_text.axis("off")

        self._setup_axes()
        self._build_artists()
        self._load_sequence(0)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Axis setup ─────────────────────────────────────────────────────────────

    def _setup_axes(self):
        for ax in (self.ax_profile, self.ax_top):
            ax.grid(True)
            ax.set_aspect("equal", adjustable="datalim")

        self.ax_profile.set_xlabel("Downrange  Z (m)")
        self.ax_profile.set_ylabel("Height  Y (m)")
        self.ax_profile.set_title("Profile View  (side-on)", color="#aaaaaa", fontsize=10)

        self.ax_top.set_xlabel("Downrange  Z (m)")
        self.ax_top.set_ylabel("Lateral  X (m)")
        self.ax_top.set_title("Top-Down View  (bird's eye)", color="#aaaaaa", fontsize=10)

    # ── Artist creation ────────────────────────────────────────────────────────

    def _build_artists(self):
        # Profile view
        self.ln_obs_p,  = self.ax_profile.plot([], [], color=_COL_OBS,  lw=2,   label="Observed")
        self.ln_ext_p,  = self.ax_profile.plot([], [], color=_COL_EXT,  lw=1.5, ls="--", label="Extrapolated")
        self.pt_ball_p, = self.ax_profile.plot([], [], "o", color=_COL_BALL, ms=8,  zorder=5)
        self.pt_land_p, = self.ax_profile.plot([], [], "x", color=_COL_LAND, ms=10, mew=2, zorder=5)
        self.pt_apex_p, = self.ax_profile.plot([], [], "^", color=_COL_APEX, ms=8,  zorder=5)

        # Top-down view
        self.ln_obs_t,  = self.ax_top.plot([], [], color=_COL_OBS,  lw=2)
        self.ln_ext_t,  = self.ax_top.plot([], [], color=_COL_EXT,  lw=1.5, ls="--")
        self.pt_ball_t, = self.ax_top.plot([], [], "o", color=_COL_BALL, ms=8,  zorder=5)
        self.pt_land_t, = self.ax_top.plot([], [], "x", color=_COL_LAND, ms=10, mew=2, zorder=5)

        self.ax_profile.legend(loc="upper left", fontsize=8, framealpha=0.3,
                               facecolor="#111111", edgecolor="#444444")

        self.title_text = self.fig.suptitle("", color="#ffffff", fontsize=12, y=1.01)
        self.info_text  = self.ax_text.text(
            0.05, 0.95, "", transform=self.ax_text.transAxes,
            va="top", ha="left", fontsize=9,
            fontfamily="monospace", color="#dddddd",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#444444", boxstyle="round,pad=0.5"),
        )

    # ── Sequence loading ───────────────────────────────────────────────────────

    def _load_sequence(self, idx: int):
        self.seq_idx  = idx % len(self.sequences)
        self.frame_idx = 0
        seq = self.sequences[self.seq_idx]
        full = seq["xyz_full"]

        # Axis limits with 10 % margin
        z_all = full[:, 2];  x_all = full[:, 0];  y_all = full[:, 1]
        zm, zM = float(z_all.min()), float(z_all.max())
        ym, yM = 0.0,              float(y_all.max())
        xm, xM = float(x_all.min()), float(x_all.max())

        def _expand(lo, hi, frac=0.1):
            pad = max((hi - lo) * frac, 1.0)
            return lo - pad, hi + pad

        self.ax_profile.set_xlim(*_expand(zm, zM))
        self.ax_profile.set_ylim(*_expand(ym, yM))
        self.ax_top.set_xlim(*_expand(zm, zM))
        self.ax_top.set_ylim(*_expand(xm, xM))

        # Find landing and apex points
        self._landing = full[full[:, 1] <= 0.01][-1] if any(full[:, 1] <= 0.01) else full[-1]
        self._apex_idx = int(np.argmax(full[:, 1]))
        self._apex    = full[self._apex_idx]

        # Static ground line on profile view
        for line in getattr(self, "_ground_lines", []):
            line.remove()
        self._ground_lines = [
            self.ax_profile.axhline(0, color="#4a4a4a", lw=1, ls="-"),
            self.ax_profile.axvline(float(full[0, 2]), color="#4a4a4a", lw=0.8, ls=":"),
        ]

        # Impact marker
        for m in getattr(self, "_impact_markers", []):
            m.remove()
        self._impact_markers = [
            self.ax_profile.plot(float(full[0, 2]), float(full[0, 1]), "|",
                                 color="#ffff00", ms=14, mew=2)[0],
            self.ax_top.plot(float(full[0, 2]), float(full[0, 0]), "|",
                             color="#ffff00", ms=14, mew=2)[0],
        ]

        # Title and info panel
        self.title_text.set_text(
            f"{seq['name']}  ({self.seq_idx + 1}/{len(self.sequences)})"
            "  ─  Space=pause  n/p=next/prev  r=restart  q=quit"
        )
        self.info_text.set_text(_format_metrics(seq["metrics"], seq["spin"]))

        # Clear all animated artists
        for art in (self.ln_obs_p, self.ln_ext_p, self.pt_ball_p, self.pt_land_p, self.pt_apex_p,
                    self.ln_obs_t, self.ln_ext_t, self.pt_ball_t, self.pt_land_t):
            art.set_data([], [])

        self.fig.canvas.draw_idle()

    # ── Animation frame ────────────────────────────────────────────────────────

    def _update(self, _frame):
        if self.paused:
            return []

        seq   = self.sequences[self.seq_idx]
        full  = seq["xyz_full"]
        n_obs = seq["n_obs"]
        i     = self.frame_idx
        n     = len(full)

        if i >= n:
            # Hold last frame for 60 ticks then auto-advance
            if i >= n + 60:
                self._load_sequence(self.seq_idx + 1)
            else:
                self.frame_idx += 1
            return []

        # Split observed vs extrapolated up to current frame
        obs_end  = min(i + 1, n_obs)
        ext_start = n_obs

        obs_z  = full[:obs_end, 2]
        obs_y  = full[:obs_end, 1]
        obs_x  = full[:obs_end, 0]

        ext_z  = full[ext_start:i + 1, 2] if i >= ext_start else np.array([])
        ext_y  = full[ext_start:i + 1, 1] if i >= ext_start else np.array([])
        ext_x  = full[ext_start:i + 1, 0] if i >= ext_start else np.array([])

        cur    = full[i]

        # Trail: only keep TRAIL_LEN most recent points for the moving segment
        trail_start = max(0, i - self.TRAIL_LEN)
        trail       = full[trail_start:i + 1]

        self.ln_obs_p.set_data(obs_z, obs_y)
        self.ln_ext_p.set_data(ext_z, ext_y)
        self.pt_ball_p.set_data([cur[2]], [cur[1]])
        self.ln_obs_t.set_data(obs_z, obs_x)
        self.ln_ext_t.set_data(ext_z, ext_x)
        self.pt_ball_t.set_data([cur[2]], [cur[0]])

        # Show landing once ball is near ground
        if cur[1] <= 0.05 or i >= n - 1:
            land = self._landing
            self.pt_land_p.set_data([land[2]], [0.0])
            self.pt_land_t.set_data([land[2]], [land[0]])
        else:
            self.pt_land_p.set_data([], [])
            self.pt_land_t.set_data([], [])

        # Show apex marker once passed
        if i >= self._apex_idx:
            apex = self._apex
            self.pt_apex_p.set_data([apex[2]], [apex[1]])
        else:
            self.pt_apex_p.set_data([], [])

        self.frame_idx += 1

        return [self.ln_obs_p, self.ln_ext_p, self.pt_ball_p, self.pt_land_p, self.pt_apex_p,
                self.ln_obs_t, self.ln_ext_t, self.pt_ball_t, self.pt_land_t]

    # ── Key handler ────────────────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.key
        if k in ("q", "escape"):
            plt.close("all")
            sys.exit(0)
        elif k in (" ", "enter"):
            self.paused = not self.paused
        elif k == "n":
            self._load_sequence(self.seq_idx + 1)
        elif k == "p":
            self._load_sequence(self.seq_idx - 1)
        elif k == "r":
            self._load_sequence(self.seq_idx)

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self):
        interval_ms = int(1000 / self.ANIM_FPS)
        self._anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animated profile + top-down golf trajectory viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir", default="outputs/test_results",
        help="Folder containing seq_XXXX_predictions.json files (default: outputs/test_results)",
    )
    parser.add_argument(
        "--seq", type=int, default=None,
        help="Show only this sequence index (e.g. 4 → seq_0004). Default: all sequences.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"Results directory not found: {results_dir}")

    seqs = find_sequences(results_dir, args.seq)
    if not seqs:
        sys.exit(f"No prediction JSON files found in {results_dir}")

    print(f"Loaded {len(seqs)} sequence(s) from {results_dir}")
    print("Controls: Space=pause  n=next  p=prev  r=restart  q=quit")

    anim = TrajectoryAnimator(seqs)
    anim.run()


if __name__ == "__main__":
    main()
