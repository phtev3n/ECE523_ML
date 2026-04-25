"""Generate ideal vs pipeline trajectory comparison plots by club category.

Simulates the standard average-case ball flight for each club type (PW,
7-iron, Driver) using published launch parameters with drag + Magnus lift.
The lift coefficient (Cl) for each club is auto-tuned via binary search to
match the published average carry distance, accounting for backspin lift
that a drag-only model would otherwise underestimate.

Club-to-sequence mapping (from shot_map.json order):
  PW     : seq_0000 – seq_0004  (IMG_9737_pw1 … IMG_9743_pw5)
  7-iron : seq_0005 – seq_0009  (IMG_9744_7i_1 … IMG_9748_7i_5)
  Driver : seq_0010 – seq_0012  (IMG_9749_Dr_1 … IMG_9758_Dr_3)

For the pipeline result, the single sequence whose predicted carry is closest
to the club's published average is selected (rather than overlaying all shots).

Outputs (in --out_dir):
  club_comparison_profile.png  — side-by-side profile view per club
  club_comparison_overlay.png  — all clubs on one axes, ideal vs best pipeline shot

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
BALL_MASS   = 0.04593        # kg
BALL_RADIUS = 0.02134        # m
BALL_AREA   = math.pi * BALL_RADIUS ** 2
AIR_DENSITY = 1.225          # kg/m³
GRAVITY     = 9.81           # m/s²
M_TO_YD     = 1.09361
YD_TO_M     = 1.0 / M_TO_YD

# ── Club definitions — average male amateur launch conditions ──────────────────
# Ball speed, launch angle, and Cd from Trackman University / USGA Distance
# Report (PGA Tour average adjusted ~15% for amateur).  Ideal carry distances
# are published amateur averages.  Cl is auto-tuned at runtime to match carry.
CLUBS: dict[str, dict] = {
    "Pitching Wedge": {
        "seq_range":      (0, 4),    # seq_0000 – seq_0004
        "speed_ms":       47.0,      # 105 mph
        "launch_deg":     24.0,
        "Cd":             0.28,
        "target_carry_yd": 120,
        "color_ideal":    "#ff8c00",
        "color_pred":     "#00d4ff",
    },
    "7-Iron": {
        "seq_range":      (5, 9),    # seq_0005 – seq_0009
        "speed_ms":       54.0,      # 121 mph
        "launch_deg":     17.0,
        "Cd":             0.25,
        "target_carry_yd": 155,
        "color_ideal":    "#ff8c00",
        "color_pred":     "#00d4ff",
    },
    "Driver": {
        "seq_range":      (10, 12),  # seq_0010 – seq_0012
        "speed_ms":       65.0,      # 145 mph
        "launch_deg":     12.0,
        "Cd":             0.21,
        "target_carry_yd": 235,
        "color_ideal":    "#ff8c00",
        "color_pred":     "#00d4ff",
    },
}

_BG     = "#0d0d0d"
_AX_BG  = "#111111"
_GRID   = "#2a2a2a"
_GROUND = "#3a3a3a"


# ── Ballistic simulation with drag + Magnus lift ───────────────────────────────

def simulate_ideal(speed_ms: float, launch_deg: float,
                   Cd: float, Cl: float = 0.0,
                   fps: float = 60.0, max_frames: int = 2400) -> np.ndarray:
    """Simulate a ballistic trajectory with aerodynamic drag and Magnus lift.

    Drag decelerates the ball proportional to v².
    Lift (backspin) is perpendicular to the velocity vector in the YZ plane,
    directed upward-forward for a ball with backspin.  As the ball descends,
    the lift component rotates to extend carry.

    Returns (N, 3) float32 array [x_lateral, y_height, z_downrange] in metres.
    All values are relative to the impact point (origin).
    """
    la   = math.radians(launch_deg)
    svz  = speed_ms * math.cos(la)
    svy  = speed_ms * math.sin(la)
    svx  = 0.0

    drag_k = 0.5 * AIR_DENSITY * Cd * BALL_AREA / BALL_MASS
    lift_k = 0.5 * AIR_DENSITY * Cl * BALL_AREA / BALL_MASS

    dt = 1.0 / fps
    pts = [[0.0, 0.0, 0.0]]
    x, y, z = 0.0, 0.0, 0.0

    for _ in range(max_frames):
        spd = math.sqrt(svx**2 + svy**2 + svz**2)
        if spd < 1e-6:
            break

        # Drag: opposes velocity
        ax = -drag_k * spd * svx
        ay = -GRAVITY - drag_k * spd * svy
        az = -drag_k * spd * svz

        # Magnus lift: perpendicular to velocity in YZ plane (backspin ↑)
        # Lift direction = rotate velocity vector 90° upward:
        #   (vy, vz) -> (-vz, vy) normalised, scaled by lift_k * v²
        lift_mag = lift_k * spd * spd
        vy_n = svy / spd
        vz_n = svz / spd
        ay  += lift_mag * vz_n    # upward component
        az  -= lift_mag * vy_n    # forward when descending, reduces carry loss

        x += svx * dt;  svx += ax * dt
        y += svy * dt;  svy += ay * dt
        z += svz * dt;  svz += az * dt

        pts.append([x, max(0.0, y), z])
        if y <= 0.0:
            break

    return np.array(pts, dtype=np.float32)


def tune_lift_for_carry(speed_ms: float, launch_deg: float, Cd: float,
                        target_carry_m: float, fps: float = 60.0,
                        tol_m: float = 0.3) -> float:
    """Binary-search the lift coefficient Cl that produces target_carry_m."""
    lo, hi = 0.0, 0.6
    for _ in range(50):
        mid = (lo + hi) / 2.0
        carry = simulate_ideal(speed_ms, launch_deg, Cd, mid, fps)[-1, 2]
        if carry < target_carry_m:
            lo = mid
        else:
            hi = mid
        if abs(carry - target_carry_m) < tol_m:
            break
    return (lo + hi) / 2.0


# ── Data loading ───────────────────────────────────────────────────────────────

def load_club_predictions(results_dir: Path,
                          seq_range: tuple[int, int]) -> list[dict]:
    """Load xyz_pred for each sequence in the range.  Returns list of dicts."""
    seqs = []
    for idx in range(seq_range[0], seq_range[1] + 1):
        p = results_dir / f"seq_{idx:04d}_predictions.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        xyz = np.asarray(data.get("xyz_pred", []), dtype=np.float32)
        if len(xyz) == 0:
            continue
        xyz = xyz - xyz[0]          # normalise to impact origin
        carry_m = float(xyz[-1, 2])
        carry_yd = carry_m * M_TO_YD
        seqs.append({"idx": idx, "xyz": xyz,
                     "carry_m": carry_m, "carry_yd": carry_yd})
    return seqs


def best_match(seqs: list[dict], target_carry_m: float) -> dict | None:
    """Return the sequence whose predicted carry is closest to target_carry_m."""
    if not seqs:
        return None
    return min(seqs, key=lambda s: abs(s["carry_m"] - target_carry_m))


# ── Axis styling helpers ───────────────────────────────────────────────────────

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


def _carry_label(ax, z_m: float, label_yd: float, color: str,
                 y_text: float = 0.5) -> None:
    """Vertical dotted line with a carry annotation in yards."""
    ax.axvline(z_m, color=color, lw=0.7, ls=":", alpha=0.55)
    ax.text(z_m, y_text, f"{label_yd:.0f} yd\n({z_m:.0f} m)",
            color=color, fontsize=6.5, ha="center", va="bottom",
            rotation=90, alpha=0.9,
            bbox=dict(facecolor=_AX_BG, edgecolor="none",
                      boxstyle="round,pad=0.15", alpha=0.6))


def _dual_axis_label(ax) -> None:
    """Add a secondary x-axis tick label row in yards."""
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) * M_TO_YD)
    ax2.tick_params(colors="#555555", labelsize=6.5, pad=1)
    ax2.set_xlabel("Downrange  Z (yd)", color="#555555", fontsize=7)
    ax2.set_facecolor(_AX_BG)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")


# ── Per-club profile panels ────────────────────────────────────────────────────

def plot_profile_comparison(results_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=_BG,
                             gridspec_kw={"wspace": 0.35})

    for ax, (club_name, cfg) in zip(axes, CLUBS.items()):
        target_m  = cfg["target_carry_yd"] * YD_TO_M
        Cl        = tune_lift_for_carry(cfg["speed_ms"], cfg["launch_deg"],
                                        cfg["Cd"], target_m)
        ideal     = simulate_ideal(cfg["speed_ms"], cfg["launch_deg"],
                                   cfg["Cd"], Cl)
        seqs      = load_club_predictions(results_dir, cfg["seq_range"])
        best      = best_match(seqs, target_m)

        _style(ax, "Downrange  Z (m)", "Height  Y (m)", club_name)

        # Best pipeline shot
        if best is not None:
            xyz = best["xyz"]
            ax.plot(xyz[:, 2], xyz[:, 1],
                    color=cfg["color_pred"], lw=2.0, ls="-",
                    label=f"Pipeline best  ({best['carry_yd']:.0f} yd / {best['carry_m']:.0f} m)")
            _carry_label(ax, best["carry_m"], best["carry_yd"],
                         cfg["color_pred"], y_text=0.4)

        # Ideal reference
        ideal_carry_m  = float(ideal[-1, 2])
        ideal_carry_yd = ideal_carry_m * M_TO_YD
        ax.plot(ideal[:, 2], ideal[:, 1],
                color=cfg["color_ideal"], lw=2.2, ls="--",
                label=f"Ideal avg  ({ideal_carry_yd:.0f} yd / {ideal_carry_m:.0f} m)")
        _carry_label(ax, ideal_carry_m, ideal_carry_yd,
                     cfg["color_ideal"], y_text=1.2)

        # Impact marker
        ax.plot(0, 0, "|", color="#ffff00", ms=14, mew=2, zorder=5)

        # Apex annotation on ideal
        apex_idx = int(np.argmax(ideal[:, 1]))
        apex_y   = float(ideal[apex_idx, 1])
        apex_z   = float(ideal[apex_idx, 2])
        ax.annotate(
            f"Apex  {apex_y:.1f} m  ({apex_y*3.281:.0f} ft)",
            xy=(apex_z, apex_y),
            xytext=(apex_z - ideal_carry_m * 0.18, apex_y + 0.8),
            color=cfg["color_ideal"], fontsize=6.5,
            arrowprops=dict(arrowstyle="->", color=cfg["color_ideal"], lw=0.8),
        )

        ax.legend(loc="upper left", fontsize=7, framealpha=0.35,
                  facecolor=_AX_BG, edgecolor="#444444", labelcolor="#cccccc")

        info = (f"Ball speed  {cfg['speed_ms']:.0f} m/s  "
                f"({cfg['speed_ms']*2.237:.0f} mph)\n"
                f"Launch      {cfg['launch_deg']:.0f}°\n"
                f"Target carry  {cfg['target_carry_yd']} yd")
        ax.text(0.98, 0.97, info, transform=ax.transAxes,
                va="top", ha="right", fontsize=6.5,
                fontfamily="monospace", color="#aaaaaa",
                bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                          boxstyle="round,pad=0.4"))

        _dual_axis_label(ax)
        print(f"  {club_name}: Cl={Cl:.4f}  "
              f"ideal={ideal_carry_m:.1f} m ({ideal_carry_yd:.0f} yd)  "
              + (f"best pipeline={best['carry_m']:.1f} m "
                 f"({best['carry_yd']:.0f} yd)" if best else "no data"))

    fig.suptitle(
        "Golf Ball Trajectory — Ideal Reference vs Best Pipeline Shot  (Profile View)",
        color="#ffffff", fontsize=11, y=1.03,
    )
    fig.savefig(out_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Profile comparison saved to {out_path}")


# ── All-clubs overlay ─────────────────────────────────────────────────────────

def plot_all_clubs_overlay(results_dir: Path, out_path: Path) -> None:
    club_colors = {
        "Pitching Wedge": "#e06c75",
        "7-Iron":         "#98c379",
        "Driver":         "#61afef",
    }

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=_BG)
    _style(ax, "Downrange  Z (m)", "Height  Y (m)",
           "All Clubs — Ideal Reference vs Best Pipeline Shot  (Profile View)")

    legend_handles = []

    for club_name, cfg in CLUBS.items():
        color     = club_colors[club_name]
        target_m  = cfg["target_carry_yd"] * YD_TO_M
        Cl        = tune_lift_for_carry(cfg["speed_ms"], cfg["launch_deg"],
                                        cfg["Cd"], target_m)
        ideal     = simulate_ideal(cfg["speed_ms"], cfg["launch_deg"],
                                   cfg["Cd"], Cl)
        seqs      = load_club_predictions(results_dir, cfg["seq_range"])
        best      = best_match(seqs, target_m)

        ideal_carry_m  = float(ideal[-1, 2])
        ideal_carry_yd = ideal_carry_m * M_TO_YD

        # Ideal trajectory
        ax.plot(ideal[:, 2], ideal[:, 1],
                color=color, lw=2.2, ls="--", alpha=0.9)
        _carry_label(ax, ideal_carry_m, ideal_carry_yd, color, y_text=0.3)

        # Best pipeline shot
        if best is not None:
            ax.plot(best["xyz"][:, 2], best["xyz"][:, 1],
                    color=color, lw=2.0, ls="-")
            _carry_label(ax, best["carry_m"], best["carry_yd"],
                         color, y_text=1.0)

        legend_handles.append(mpatches.Patch(
            color=color,
            label=(f"{club_name}:  "
                   f"ideal {ideal_carry_yd:.0f} yd  /  "
                   + (f"pipeline {best['carry_yd']:.0f} yd" if best else "no data"))
        ))

    ax.plot(0, 0, "|", color="#ffff00", ms=16, mew=2.5, zorder=5)

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.35, facecolor=_AX_BG, edgecolor="#444444",
              labelcolor="#cccccc")

    ax.text(0.98, 0.97,
            "Dashed  =  ideal simulation (drag + Magnus lift)\n"
            "Solid    =  best pipeline predicted shot",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=7, fontfamily="monospace", color="#888888",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                      boxstyle="round,pad=0.4"))

    _dual_axis_label(ax)

    fig.suptitle(
        "Golf Ball Trajectory Comparison by Club  —  Profile View",
        color="#ffffff", fontsize=12, y=1.03,
    )
    fig.savefig(out_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay comparison saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ideal club trajectories to pipeline predictions"
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

    print("Tuning lift coefficients to match target carry distances...")
    plot_profile_comparison(results_dir, out_dir / "club_comparison_profile.png")
    plot_all_clubs_overlay(results_dir,  out_dir / "club_comparison_overlay.png")
    print(f"\nDone. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
