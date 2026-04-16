"""Evaluate the full end-to-end golf ball tracking pipeline on a dataset.

Runs the complete GolfBallTrackingPipeline (detector → Kalman filter →
TrajectoryLifter) on every sequence in the dataset and reports:
  - 2D tracking metrics (RMSE, visibility F1, smoothness)
  - 3D reconstruction metrics (RMSE, carry error, apex error)
  - Ball flight metrics (launch angle, carry, apex, descent, ToF)
  - Spin estimation (model spin_head or physics-fitting fallback)

Output files per sequence (in --out_dir):
  seq_XXXX_predictions.json  — all predictions + metrics for that sequence
  seq_XXXX_overlay.mp4       — tracer + HUD overlay video (if --save_video)

Summary output:
  metrics.csv   — per-sequence metric table
  summary.json  — mean of each metric across all sequences

Spin estimation strategy
------------------------
The model's spin_head output is used when its backspin magnitude exceeds
100 rpm (i.e. a physically plausible shot).  Near-zero values indicate the
LSTM lacked sufficient clip length to observe Magnus curvature, so the
physics-fitting fallback (estimate_spin_from_trajectory) is invoked instead.
That fallback requires clips ≥ 90 frames (1.5 s at 60 fps); for shorter clips
both estimators report 0 and spin rows are omitted from the HUD.

Usage
-----
python scripts/test_pipeline.py \\
    --config configs/pipeline.yaml \\
    --dataset_root demo_dataset \\
    --detector_ckpt outputs/detector_best.pt \\
    --trajectory_ckpt outputs/trajectory_best.pt \\
    --save_video
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import cv2
import numpy as np
import torch

from golf_tracer.data.real_dataset import RealGolfSequenceDataset
from golf_tracer.models.detector import MultiScaleBallDetector
from golf_tracer.models.trajectory_lifter import TrajectoryLifter
from golf_tracer.tracking.pipeline import GolfBallTrackingPipeline
from golf_tracer.utils.config import load_config
from golf_tracer.utils.geometry import (
    apex_height,
    compute_ball_metrics,
    estimate_carry_from_xyz,
    estimate_spin_from_trajectory,
)
from golf_tracer.utils.io import ensure_dir, write_json
from golf_tracer.utils.metrics import binary_f1, rmse, smoothness
from golf_tracer.utils.render import draw_metrics_panel, draw_tracer, render_video
from golf_tracer.utils.train import resolve_device


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device)
    except TypeError:
        return torch.load(path, map_location=device, weights_only=False)


def load_models(detector_ckpt: str, trajectory_ckpt: str, device: torch.device):
    det_ckpt = safe_torch_load(detector_ckpt, device)
    traj_ckpt = safe_torch_load(trajectory_ckpt, device)

    det_backbone = det_ckpt["config"]["model"]["backbone"]
    detector = MultiScaleBallDetector(det_backbone).to(device)
    detector.load_state_dict(det_ckpt["model_state"])
    detector.eval()

    traj_cfg = traj_ckpt["config"]["model"]
    trajectory_model = TrajectoryLifter(
        input_size=5,
        hidden_size=traj_cfg["hidden_size"],
        num_layers=traj_cfg["num_layers"],
    ).to(device)
    trajectory_model.load_state_dict(traj_ckpt["model_state"])
    trajectory_model.eval()

    return detector, trajectory_model


def has_bad_values(x) -> bool:
    arr = np.asarray(x)
    return np.isnan(arr).any() or np.isinf(arr).any()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--detector_ckpt", required=True)
    parser.add_argument("--trajectory_ckpt", required=True)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--out_dir", default="outputs/test_results")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vis_threshold", type=float, default=0.35)
    parser.add_argument("--use_gt_uv", action="store_true",
                        help="Bypass detector entirely — feed ground-truth UV to the LSTM. "
                             "Use this to measure the LSTM ceiling performance independently "
                             "of detector quality.")
    parser.add_argument("--classical_detector", action="store_true",
                        help="Use background-subtraction + blob-size detection instead of "
                             "the learned detector.  No domain gap; works best for a static "
                             "camera where the ball is the only small fast-moving object.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    dataset = RealGolfSequenceDataset(
        args.dataset_root,
        sequence_length=cfg["sequence_length"],
        mode="trajectory",
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    detector, trajectory_model = load_models(args.detector_ckpt, args.trajectory_ckpt, device)
    pipeline = GolfBallTrackingPipeline(detector, trajectory_model, cfg, device)

    out_dir = ensure_dir(args.out_dir)
    rows = []

    max_cfg = cfg.get("max_test_sequences", len(dataset))
    max_n = min(len(dataset), max_cfg)
    if args.max_sequences is not None:
        max_n = min(max_n, args.max_sequences)

    with torch.no_grad():
        for idx in range(max_n):
            sample = dataset[idx]
            frames = sample["frames"]
            camera = sample["camera"]
            gt_uv = sample["uv"].cpu().numpy()
            gt_xyz = sample["xyz"].cpu().numpy()
            gt_vis = sample["visible"].cpu().numpy()

            if args.use_gt_uv:
                result = pipeline.run_sequence(
                    frames, camera,
                    precomputed_uv=gt_uv,
                    precomputed_vis=gt_vis,
                )
            elif args.classical_detector:
                result = pipeline.run_sequence(frames, camera, use_classical=True)
            else:
                result = pipeline.run_sequence(frames, camera)

            if args.verbose:
                print(f"seq {idx}")
                print(
                    "visible_prob min/max/mean:",
                    float(np.min(result.visible_prob)),
                    float(np.max(result.visible_prob)),
                    float(np.mean(result.visible_prob)),
                )
                print("gt visible mean:", float(np.mean(gt_vis)))
                print("first 5 measured_uv:", result.measured_uv[:5])
                print("first 5 gt_uv:", gt_uv[:5])

            if has_bad_values(result.measured_uv) or has_bad_values(result.filtered_uv) or has_bad_values(result.xyz_pred):
                print(f"Warning: NaN/Inf detected in sequence {idx}")

            # ---- Ball flight metrics from predicted 3D trajectory ----
            # Use the sequence fps stored in the dataset metadata; fall back to
            # the Kalman dt so at least carry/ToF are computed consistently.
            fps_seq = float(sample.get("fps", 1.0 / cfg["kalman"]["dt"]))
            ball_metrics = compute_ball_metrics(result.xyz_pred, fps_seq)

            # Spin: use model spin_head if its backspin prediction is physiologically
            # plausible (> 100 rpm); otherwise fall back to physics fitting on xyz_pred.
            spin_model = result.spin_pred
            if float(np.abs(spin_model[0])) > 100.0:
                spin_display = {
                    "backspin_rpm": float(spin_model[0]),
                    "sidespin_rpm": float(spin_model[1]),
                }
            else:
                try:
                    spin_display = estimate_spin_from_trajectory(result.xyz_pred, fps_seq)
                except Exception:
                    spin_display = None

            row = {
                "sequence": idx,
                "rmse_2d_measured": rmse(result.measured_uv, gt_uv, gt_vis > 0.5),
                "rmse_2d_filtered": rmse(result.filtered_uv, gt_uv, gt_vis > 0.5),
                "rmse_3d": rmse(result.xyz_pred, gt_xyz),
                "visibility_f1": binary_f1(result.visible_prob, gt_vis, thresh=args.vis_threshold),
                "smoothness_filtered": smoothness(result.filtered_uv),
                "carry_err_m": abs(
                    estimate_carry_from_xyz(result.xyz_pred) - estimate_carry_from_xyz(gt_xyz)
                ),
                "apex_err_m": abs(apex_height(result.xyz_pred) - apex_height(gt_xyz)),
                "visible_prob_mean": float(np.mean(result.visible_prob)),
                "visible_prob_max": float(np.max(result.visible_prob)),
                "gt_visible_rate": float(np.mean(gt_vis)),
                **{f"ball_{k}": v for k, v in ball_metrics.items()},
                "spin_backspin_rpm": float(spin_display["backspin_rpm"]) if spin_display else 0.0,
                "spin_sidespin_rpm": float(spin_display["sidespin_rpm"]) if spin_display else 0.0,
            }
            rows.append(row)

            pred_json = {
                "measured_uv": np.asarray(result.measured_uv).tolist(),
                "filtered_uv": np.asarray(result.filtered_uv).tolist(),
                "xyz_pred": np.asarray(result.xyz_pred).tolist(),
                "uv_reprojected": np.asarray(result.uv_reprojected).tolist(),
                "visible_prob": np.asarray(result.visible_prob).tolist(),
                "ball_metrics": ball_metrics,
                "spin": spin_display,
                "metrics": row,
            }
            write_json(out_dir / f"seq_{idx:04d}_predictions.json", pred_json)

            if args.save_video:
                frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
                frames_np = np.clip(frames_np, 0.0, 1.0)
                frames_np = (frames_np * 255.0).astype(np.uint8)

                # First frame from which the Kalman filter was validly seeded.
                # Points before this index are raw (unsmoothed) detector outputs
                # on low-confidence frames and must not be drawn as part of the
                # tracer — they are scattered false positives, not ball positions.
                kf_start = max(0, result.kf_init_frame)

                vis_frames = []
                for t, frame in enumerate(frames_np):
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if cfg["render"]["draw_measurements"]:
                        draw_tracer(bgr, result.measured_uv[: t + 1], color=(255, 255, 0), thickness=1)
                    if cfg["render"]["draw_filtered"] and t >= kf_start:
                        draw_tracer(
                            bgr,
                            result.filtered_uv[kf_start: t + 1],
                            color=(0, 255, 255),
                            thickness=cfg["render"]["tracer_thickness"],
                        )
                    if cfg["render"]["draw_reprojected"]:
                        draw_tracer(bgr, result.uv_reprojected[: t + 1], color=(0, 100, 255), thickness=1)
                    if cfg["render"].get("draw_metrics", True) and ball_metrics:
                        draw_metrics_panel(bgr, ball_metrics, spin=spin_display)
                    vis_frames.append(bgr)

                render_video(vis_frames, out_dir / f"seq_{idx:04d}_overlay.mp4", fps=fps_seq)

    metrics_path = out_dir / "metrics.csv"
    if rows:
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            k: float(np.mean([r[k] for r in rows]))
            for k in rows[0].keys()
            if k != "sequence"
        }
        write_json(out_dir / "summary.json", summary)

        print("Average metrics:")
        for k, v in summary.items():
            print(f"  {k}: {v:.6f}")
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved summary to {out_dir / 'summary.json'}")
    else:
        print("No rows were generated; nothing to save.")

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()