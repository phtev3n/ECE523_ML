"""Generate static 3D trajectory comparison plots for submission.

For each sequence, reads:
  - seq_XXXX_predictions.json  (xyz_pred from LSTM + ball metrics)
  - <dataset_root>/seq_XXXX/annotations.json  (GT xyz + visibility)

Produces per-sequence PNG figures (profile + top-down, predicted vs GT)
and a summary grid figure showing all shots on one page.

Runs headless (matplotlib Agg backend) — safe on HPC with no display.

Usage
-----
python scripts/plot_trajectories.py \
    --results_dir outputs/demo_videos \
    --dataset_root real_data_work/dataset \
    --out_dir outputs/trajectory_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

G = 9.81  # m/s²

# ── Colour palette ─────────────────────────────────────────────────────────────
_BG      = "#0d0d0d"
_AX_BG   = "#111111"
_GRID    = "#2a2a2a"
_PRED    = "#00d4ff"   # cyan  — LSTM predicted
_GT      = "#ff8c00"   # amber — ground truth
_EXTRAP  = "#44ff88"   # green — ballistic extrapolation
_GROUND  = "#4a4a4a"


# ── Physics helpers ────────────────────────────────────────────────────────────

def extrapolate_to_landing(xyz: np.ndarray, fps: float, max_extra: int = 600) -> np.ndarray:
    """Forward-integrate a drag-free ballistic arc from the last observed point."""
    if len(xyz) < 2:
        return xyz
    dt = 1.0 / fps
    n0 = min(4, len(xyz) - 1)
    v0 = (xyz[n0] - xyz[0]) / (n0 * dt)
    vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
    pts = [xyz[-1].copy()]
    x, y, z = float(xyz[-1, 0]), float(xyz[-1, 1]), float(xyz[-1, 2])
    for _ in range(max_extra):
        x += vx * dt
        y += vy * dt
        z += vz * dt
        vy -= G * dt
        pts.append([x, max(0.0, y), z])
        if y <= 0.0:
            break
    return np.array(pts, dtype=np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sequence(pred_path: Path, dataset_root: Path) -> dict | None:
    """Load predicted and GT trajectory for one sequence."""
    with open(pred_path) as f:
        pred = json.load(f)

    xyz_pred = np.asarray(pred.get("xyz_pred", []), dtype=np.float32)
    if len(xyz_pred) == 0:
        return None

    # GT xyz from annotations
    seq_name = pred_path.stem.replace("_predictions", "")
    ann_path = dataset_root / seq_name / "annotations.json"
    xyz_gt = np.zeros_like(xyz_pred)
    vis_gt = np.zeros(len(xyz_pred), dtype=bool)
    fps = 60.0
    if ann_path.exists():
        with open(ann_path) as f:
            ann = json.load(f)
        fps = float(ann.get("fps", 60.0))
        frames = ann.get("frames", [])
        n = min(len(frames), len(xyz_pred))
        xyz_gt[:n] = [[fr["xyz"][0], fr["xyz"][1], fr["xyz"][2]] for fr in frames[:n]]
        vis_gt[:n] = [bool(fr["visible"]) for fr in frames[:n]]

    metrics = pred.get("metrics", {})
    ball_metrics = pred.get("ball_metrics", {})

    return {
        "name":      seq_name,
        "xyz_pred":  xyz_pred,
        "xyz_gt":    xyz_gt,
        "vis_gt":    vis_gt,
        "extrap":    extrapolate_to_landing(xyz_pred, fps),
        "fps":       fps,
        "metrics":   metrics,
        "ball":      ball_metrics,
    }


def find_sequences(results_dir: Path, dataset_root: Path) -> list[dict]:
    seqs = []
    for p in sorted(results_dir.glob("seq_*_predictions.json")):
        s = load_sequence(p, dataset_root)
        if s is not None:
            seqs.append(s)
    return seqs


# ── Axis styling ───────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.set_title(title, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _metrics_text(seq: dict) -> str:
    b = seq["ball"]
    m = seq["metrics"]
    lines = []
    if "ball_speed_ms" in b:
        lines.append(f"Speed   {b['ball_speed_ms']:.1f} m/s  ({b['ball_speed_ms']*2.237:.0f} mph)")
    if "launch_angle_deg" in b:
        lines.append(f"Launch  {b['launch_angle_deg']:.1f}°")
    if "carry_m" in b:
        lines.append(f"Carry   {b['carry_m']:.0f} m  ({b['carry_m']*1.094:.0f} yd)")
    if "apex_m" in b:
        lines.append(f"Apex    {b['apex_m']:.1f} m  ({b['apex_m']*3.281:.0f} ft)")
    lines.append("─" * 22)
    if "rmse_2d_measured" in m:
        lines.append(f"RMSE 2D  {m['rmse_2d_measured']:.1f} px")
    if "rmse_3d" in m:
        lines.append(f"RMSE 3D  {m['rmse_3d']:.2f} m")
    if "visibility_f1" in m:
        lines.append(f"Vis F1   {m['visibility_f1']:.3f}")
    return "\n".join(lines)


# ── Per-sequence figure ────────────────────────────────────────────────────────

def plot_sequence(seq: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(13, 5), facecolor=_BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[5, 5, 2],
                           left=0.06, right=0.97, top=0.88, bottom=0.12, wspace=0.3)
    ax_prof = fig.add_subplot(gs[0])
    ax_top  = fig.add_subplot(gs[1])
    ax_info = fig.add_subplot(gs[2])
    ax_info.axis("off")

    _style_ax(ax_prof, "Downrange  Z (m)", "Height  Y (m)",  "Profile View (side-on)")
    _style_ax(ax_top,  "Downrange  Z (m)", "Lateral  X (m)", "Top-Down View (bird's eye)")

    pred = seq["xyz_pred"]
    gt   = seq["xyz_gt"]
    vis  = seq["vis_gt"]
    ext  = seq["extrap"]

    # Ground line
    ax_prof.axhline(0, color=_GROUND, lw=0.8, ls="-")

    # GT trajectory (visible frames only)
    if vis.any():
        ax_prof.plot(gt[vis, 2], gt[vis, 1], "o", color=_GT,   ms=4, label="GT (visible)", zorder=4)
        ax_top.plot( gt[vis, 2], gt[vis, 0], "o", color=_GT,   ms=4, zorder=4)

    # Predicted observed arc
    ax_prof.plot(pred[:, 2], pred[:, 1], "-",  color=_PRED, lw=2,   label="Predicted", zorder=3)
    ax_top.plot( pred[:, 2], pred[:, 0], "-",  color=_PRED, lw=2,   zorder=3)

    # Ballistic extrapolation
    if len(ext) > 1:
        ax_prof.plot(ext[:, 2], ext[:, 1], "--", color=_EXTRAP, lw=1.2, label="Extrapolated", zorder=2)
        ax_top.plot( ext[:, 2], ext[:, 0], "--", color=_EXTRAP, lw=1.2, zorder=2)

    # Impact marker
    ax_prof.plot(pred[0, 2], pred[0, 1], "|", color="#ffff00", ms=14, mew=2, zorder=5)
    ax_top.plot( pred[0, 2], pred[0, 0], "|", color="#ffff00", ms=14, mew=2, zorder=5)

    # Legend
    ax_prof.legend(loc="upper left", fontsize=7, framealpha=0.3,
                   facecolor=_AX_BG, edgecolor="#444444",
                   labelcolor="#cccccc")

    # Metrics panel
    ax_info.text(0.05, 0.95, _metrics_text(seq),
                 transform=ax_info.transAxes, va="top", ha="left",
                 fontsize=8, fontfamily="monospace", color="#dddddd",
                 bbox=dict(facecolor="#1a1a1a", edgecolor="#444444", boxstyle="round,pad=0.5"))

    fig.suptitle(seq["name"], color="#ffffff", fontsize=11, y=0.97)
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)


# ── Summary grid figure ────────────────────────────────────────────────────────

def plot_summary_grid(seqs: list[dict], out_path: Path) -> None:
    """Profile-view trajectories for all shots on a single figure."""
    n = len(seqs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                              facecolor=_BG, squeeze=False)

    for i, seq in enumerate(seqs):
        ax = axes[i // ncols][i % ncols]
        _style_ax(ax, "Z (m)", "Y (m)", seq["name"])

        pred = seq["xyz_pred"]
        gt   = seq["xyz_gt"]
        vis  = seq["vis_gt"]
        ext  = seq["extrap"]

        ax.axhline(0, color=_GROUND, lw=0.6)
        if vis.any():
            ax.plot(gt[vis, 2], gt[vis, 1], "o", color=_GT,   ms=3, label="GT")
        ax.plot(pred[:, 2], pred[:, 1], "-",  color=_PRED, lw=1.5, label="Pred")
        if len(ext) > 1:
            ax.plot(ext[:, 2],  ext[:, 1],  "--", color=_EXTRAP, lw=1.0)

        b = seq["ball"]
        subtitle = ""
        if "carry_m" in b:
            subtitle += f"Carry {b['carry_m']:.0f}m"
        if "launch_angle_deg" in b:
            subtitle += f"  •  {b['launch_angle_deg']:.0f}°"
        ax.set_title(f"{seq['name']}\n{subtitle}", color="#aaaaaa", fontsize=7)

        if i == 0:
            ax.legend(fontsize=6, framealpha=0.3, facecolor=_AX_BG,
                      edgecolor="#444444", labelcolor="#cccccc")

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Golf Ball Trajectory — Predicted vs Ground Truth",
                 color="#ffffff", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary grid saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs GT 3D trajectories for submission")
    parser.add_argument("--results_dir",  required=True, help="Folder with seq_*_predictions.json")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with seq_XXXX/annotations.json")
    parser.add_argument("--out_dir",      default="outputs/trajectory_plots")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    dataset_root = Path(args.dataset_root)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = find_sequences(results_dir, dataset_root)
    if not seqs:
        sys.exit(f"No seq_*_predictions.json files found in {results_dir}")

    print(f"Plotting {len(seqs)} sequences → {out_dir}")

    for seq in seqs:
        out_path = out_dir / f"{seq['name']}_trajectory.png"
        plot_sequence(seq, out_path)
        print(f"  {seq['name']}  carry={seq['ball'].get('carry_m', 0):.0f}m  → {out_path.name}")

    plot_summary_grid(seqs, out_dir / "summary_grid.png")
    print(f"\nDone. {len(seqs)} sequence plots + summary_grid.png in {out_dir}")


if __name__ == "__main__":
    main()
