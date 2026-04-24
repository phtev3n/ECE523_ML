"""Generate ideal vs pipeline trajectory comparison plots by club category.

Simulates the standard average-case ball flight for each club type (PW,
7-iron, Driver) using published launch parameters, then overlays the
pipeline's predicted trajectories for the same club groupings from the
test results.

Club-to-sequence mapping (from shot_map.json order):
  PW     : seq_0000 – seq_0004  (IMG_9737_pw1 … IMG_9743_pw5)
  7-iron : seq_0005 – seq_0009  (IMG_9744_7i_1 … IMG_9748_7i_5)
  Driver : seq_0010 – seq_0012  (IMG_9749_Dr_1 … IMG_9758_Dr_3)

Outputs (in --out_dir):
  club_comparison_profile.png   — side-by-side profile view per club
  club_comparison_topdown.png   — side-by-side top-down view per club
  club_comparison_overlay.png   — all clubs on one axes, ideal vs predicted

Runs headless (matplotlib Agg) — safe on HPC with no display.

Usage
-----
python scripts/plot_club_comparison.py \
    --results_dir outputs/demo_videos \
    --out_dir     outputs/trajectory_plots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Physics constants ──────────────────────────────────────────────────────────
BALL_MASS   = 0.04593   # kg
BALL_RADIUS = 0.02134   # m
BALL_AREA   = math.pi * BALL_RADIUS ** 2
AIR_DENSITY = 1.225     # kg/m³
GRAVITY     = 9.81      # m/s²

# ── Club definitions — average male amateur launch conditions ──────────────────
# Sources: Trackman University, USGA Distance Report, PGA Tour averages adjusted
# downward ~15% for amateur.
CLUBS: dict[str, dict] = {
    "Pitching Wedge": {
        "seq_range":      (0, 4),       # seq_0000 – seq_0004 inclusive
        "speed_ms":       47.0,         # ~105 mph
        "launch_deg":     24.0,
        "Cd":             0.28,         # higher drag (high spin)
        "ideal_carry_yd": 120,
        "color_ideal":    "#ff8c00",    # amber
        "color_pred":     "#00d4ff",    # cyan
        "color_gt":       "#aaaaaa",    # grey dots
    },
    "7-Iron": {
        "seq_range":      (5, 9),       # seq_0005 – seq_0009
        "speed_ms":       54.0,         # ~121 mph
        "launch_deg":     17.0,
        "Cd":             0.25,
        "ideal_carry_yd": 155,
        "color_ideal":    "#ff8c00",
        "color_pred":     "#00d4ff",
        "color_gt":       "#aaaaaa",
    },
    "Driver": {
        "seq_range":      (10, 12),     # seq_0010 – seq_0012
        "speed_ms":       65.0,         # ~145 mph
        "launch_deg":     12.0,
        "Cd":             0.21,         # lower drag (lower spin)
        "ideal_carry_yd": 235,
        "color_ideal":    "#ff8c00",
        "color_pred":     "#00d4ff",
        "color_gt":       "#aaaaaa",
    },
}

# ── Colour palette ─────────────────────────────────────────────────────────────
_BG    = "#0d0d0d"
_AX_BG = "#111111"
_GRID  = "#2a2a2a"
_GROUND = "#3a3a3a"


# ── Ballistic simulation ───────────────────────────────────────────────────────

def simulate_ideal(speed_ms: float, launch_deg: float, Cd: float,
                   fps: float = 60.0, max_frames: int = 1200) -> np.ndarray:
    """Simulate a drag-only ballistic trajectory starting at the origin.

    Returns (N, 3) array of [x, y, z] positions in metres, where z is
    downrange distance and y is height.  x (lateral) is zero throughout
    for a straight shot.
    """
    la  = math.radians(launch_deg)
    vz  = speed_ms * math.cos(la)   # downrange
    vy  = speed_ms * math.sin(la)   # upward
    drag_k = 0.5 * AIR_DENSITY * Cd * BALL_AREA / BALL_MASS

    dt    = 1.0 / fps
    pts   = [[0.0, 0.0, 0.0]]
    x, y, z = 0.0, 0.0, 0.0
    svx, svy, svz = 0.0, vy, vz

    for _ in range(max_frames):
        spd = math.sqrt(svx**2 + svy**2 + svz**2)
        ax  = -drag_k * spd * svx
        ay  = -GRAVITY - drag_k * spd * svy
        az  = -drag_k * spd * svz

        x  += svx * dt;  svx += ax * dt
        y  += svy * dt;  svy += ay * dt
        z  += svz * dt;  svz += az * dt

        pts.append([x, max(0.0, y), z])
        if y <= 0.0:
            break

    return np.array(pts, dtype=np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_club_predictions(results_dir: Path, seq_range: tuple[int, int]) -> list[np.ndarray]:
    """Return list of xyz_pred arrays for sequences in [seq_range[0], seq_range[1]]."""
    trajs = []
    for idx in range(seq_range[0], seq_range[1] + 1):
        p = results_dir / f"seq_{idx:04d}_predictions.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        xyz = np.asarray(data.get("xyz_pred", []), dtype=np.float32)
        if len(xyz) == 0:
            continue
        # Normalise: shift so impact point is at origin (z=0, y=0)
        xyz = xyz - xyz[0]
        trajs.append(xyz)
    return trajs


# ── Axis styling ───────────────────────────────────────────────────────────────

def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, linewidth=0.5)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.set_title(title, color="#cccccc", fontsize=9, pad=4)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.axhline(0, color=_GROUND, lw=0.8)


def _annotate_carry(ax, xyz: np.ndarray, label: str, color: str) -> None:
    """Mark the landing point (last y≈0) with a carry annotation."""
    landing_z = float(xyz[-1, 2])
    carry_yd  = landing_z * 1.09361
    ax.axvline(landing_z, color=color, lw=0.6, ls=":", alpha=0.5)
    ax.text(landing_z, 0.4, f"{carry_yd:.0f} yd",
            color=color, fontsize=6.5, ha="center", va="bottom",
            rotation=90, alpha=0.85)


# ── Main plot: per-club profile panels ────────────────────────────────────────

def plot_profile_comparison(results_dir: Path, out_path: Path) -> None:
    """Three-panel profile view — one panel per club category."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=_BG,
                             gridspec_kw={"wspace": 0.32})

    for ax, (club_name, cfg) in zip(axes, CLUBS.items()):
        ideal = simulate_ideal(cfg["speed_ms"], cfg["launch_deg"], cfg["Cd"])
        preds = load_club_predictions(results_dir, cfg["seq_range"])

        _style(ax, "Downrange  Z (m)", "Height  Y (m)", club_name)

        # Pipeline predicted trajectories (individual, semi-transparent)
        for i, xyz in enumerate(preds):
            ax.plot(xyz[:, 2], xyz[:, 1],
                    color=cfg["color_pred"], lw=1.2, alpha=0.45,
                    label="Pipeline predicted" if i == 0 else None)

        # Ideal reference trajectory
        ax.plot(ideal[:, 2], ideal[:, 1],
                color=cfg["color_ideal"], lw=2.2, ls="--",
                label=f"Ideal  ({cfg['ideal_carry_yd']} yd avg)")

        # Carry annotations
        _annotate_carry(ax, ideal, "Ideal", cfg["color_ideal"])
        if preds:
            # Use the predicted trajectory with the median carry
            carries = [xyz[-1, 2] for xyz in preds]
            median_xyz = preds[int(np.argsort(carries)[len(carries) // 2])]
            _annotate_carry(ax, median_xyz, "Pipeline", cfg["color_pred"])

        # Impact marker
        ax.plot(0, 0, "|", color="#ffff00", ms=14, mew=2, zorder=5)

        # Apex annotation for ideal
        apex_idx = int(np.argmax(ideal[:, 1]))
        ax.annotate(
            f"Apex {ideal[apex_idx, 1]:.1f} m",
            xy=(ideal[apex_idx, 2], ideal[apex_idx, 1]),
            xytext=(ideal[apex_idx, 2] - ideal[-1, 2] * 0.15,
                    ideal[apex_idx, 1] + 1.5),
            color=cfg["color_ideal"], fontsize=6.5,
            arrowprops=dict(arrowstyle="->", color=cfg["color_ideal"],
                            lw=0.8),
        )

        # Legend
        ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
                  facecolor=_AX_BG, edgecolor="#444444",
                  labelcolor="#cccccc")

        # Parameter box
        params = (f"Speed  {cfg['speed_ms']:.0f} m/s"
                  f"  ({cfg['speed_ms']*2.237:.0f} mph)\n"
                  f"Launch {cfg['launch_deg']:.0f}°\n"
                  f"n={cfg['seq_range'][1]-cfg['seq_range'][0]+1} shots")
        ax.text(0.98, 0.97, params,
                transform=ax.transAxes, va="top", ha="right",
                fontsize=6.5, fontfamily="monospace", color="#aaaaaa",
                bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                          boxstyle="round,pad=0.4"))

    fig.suptitle(
        "Golf Ball Trajectory — Ideal Reference vs Pipeline Prediction  (Profile View)",
        color="#ffffff", fontsize=11, y=1.01,
    )
    fig.savefig(out_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Profile comparison saved to {out_path}")


# ── Overlay plot: all clubs on one axes ───────────────────────────────────────

def plot_all_clubs_overlay(results_dir: Path, out_path: Path) -> None:
    """Single profile axes with all three clubs, ideal vs predicted."""
    club_colors = {
        "Pitching Wedge": "#e06c75",   # red
        "7-Iron":         "#98c379",   # green
        "Driver":         "#61afef",   # blue
    }

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
    _style(ax, "Downrange  Z (m)", "Height  Y (m)",
           "All Clubs — Ideal Reference vs Pipeline Prediction")

    legend_handles = []

    for club_name, cfg in CLUBS.items():
        color = club_colors[club_name]
        ideal = simulate_ideal(cfg["speed_ms"], cfg["launch_deg"], cfg["Cd"])
        preds = load_club_predictions(results_dir, cfg["seq_range"])

        # Individual predicted arcs
        for i, xyz in enumerate(preds):
            ax.plot(xyz[:, 2], xyz[:, 1],
                    color=color, lw=1.0, alpha=0.30)

        # Ideal reference
        line, = ax.plot(ideal[:, 2], ideal[:, 1],
                        color=color, lw=2.2, ls="--")

        # Average predicted (median carry)
        if preds:
            carries = [xyz[-1, 2] for xyz in preds]
            med_xyz = preds[int(np.argsort(carries)[len(carries) // 2])]
            ax.plot(med_xyz[:, 2], med_xyz[:, 1],
                    color=color, lw=1.8, ls="-")

        _annotate_carry(ax, ideal, club_name, color)

        legend_handles.append(
            mpatches.Patch(color=color, label=f"{club_name}  (-- ideal / — pipeline)")
        )

    ax.plot(0, 0, "|", color="#ffff00", ms=16, mew=2.5, zorder=5)

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.3, facecolor=_AX_BG, edgecolor="#444444",
              labelcolor="#cccccc")

    # Annotation explaining line styles
    ax.text(0.98, 0.97,
            "Dashed  =  ideal simulation\nSolid     =  pipeline predicted (median)",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=7, fontfamily="monospace", color="#888888",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Golf Ball Trajectory Comparison by Club  —  Profile View",
                 color="#ffffff", fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay comparison saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ideal reference trajectories to pipeline predictions by club"
    )
    parser.add_argument("--results_dir", required=True,
                        help="Folder containing seq_*_predictions.json")
    parser.add_argument("--out_dir", default="outputs/trajectory_plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        sys.exit(f"Results directory not found: {results_dir}")

    plot_profile_comparison(results_dir, out_dir / "club_comparison_profile.png")
    plot_all_clubs_overlay(results_dir,  out_dir / "club_comparison_overlay.png")

    print(f"\nDone. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
