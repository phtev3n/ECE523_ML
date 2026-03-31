from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import csv
import cv2
import numpy as np
import torch

from golf_tracer.data.real_dataset import RealGolfSequenceDataset
from golf_tracer.models.detector import MultiScaleBallDetector
from golf_tracer.models.trajectory_lifter import TrajectoryLifter
from golf_tracer.tracking.pipeline import GolfBallTrackingPipeline
from golf_tracer.utils.config import load_config
from golf_tracer.utils.geometry import apex_height, estimate_carry_from_xyz
from golf_tracer.utils.io import ensure_dir, write_json
from golf_tracer.utils.metrics import binary_f1, rmse, smoothness
from golf_tracer.utils.render import draw_tracer, render_video
from golf_tracer.utils.train import resolve_device


def load_models(detector_ckpt: str, trajectory_ckpt: str, device: torch.device):
    det_ckpt = torch.load(detector_ckpt, map_location=device)
    traj_ckpt = torch.load(trajectory_ckpt, map_location=device)

    det_backbone = det_ckpt["config"]["model"]["backbone"]
    detector = MultiScaleBallDetector(det_backbone).to(device)
    detector.load_state_dict(det_ckpt["model_state"])
    detector.eval()

    traj_cfg = traj_ckpt["config"]["model"]
    trajectory_model = TrajectoryLifter(input_size=5, hidden_size=traj_cfg["hidden_size"], num_layers=traj_cfg["num_layers"]).to(device)
    trajectory_model.load_state_dict(traj_ckpt["model_state"])
    trajectory_model.eval()
    return detector, trajectory_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--detector_ckpt", required=True)
    parser.add_argument("--trajectory_ckpt", required=True)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    dataset = RealGolfSequenceDataset(args.dataset_root, sequence_length=cfg["sequence_length"], mode="trajectory")
    detector, trajectory_model = load_models(args.detector_ckpt, args.trajectory_ckpt, device)
    pipeline = GolfBallTrackingPipeline(detector, trajectory_model, cfg, device)

    out_dir = ensure_dir("outputs/test_results")
    rows = []

    max_n = min(len(dataset), cfg["max_test_sequences"])
    for idx in range(max_n):
        sample = dataset[idx]
        frames = sample["frames"]
        camera = sample["camera"]
        gt_uv = sample["uv"].numpy()
        gt_xyz = sample["xyz"].numpy()
        gt_vis = sample["visible"].numpy()
        result = pipeline.run_sequence(frames, camera)

        row = {
            "sequence": idx,
            "rmse_2d_measured": rmse(result.measured_uv, gt_uv, gt_vis > 0.5),
            "rmse_2d_filtered": rmse(result.filtered_uv, gt_uv, gt_vis > 0.5),
            "rmse_3d": rmse(result.xyz_pred, gt_xyz),
            "visibility_f1": binary_f1(result.visible_prob, gt_vis),
            "smoothness_filtered": smoothness(result.filtered_uv),
            "carry_err_m": abs(estimate_carry_from_xyz(result.xyz_pred) - estimate_carry_from_xyz(gt_xyz)),
            "apex_err_m": abs(apex_height(result.xyz_pred) - apex_height(gt_xyz)),
        }
        rows.append(row)

        pred_json = {
            "measured_uv": result.measured_uv.tolist(),
            "filtered_uv": result.filtered_uv.tolist(),
            "xyz_pred": result.xyz_pred.tolist(),
            "uv_reprojected": result.uv_reprojected.tolist(),
            "visible_prob": result.visible_prob.tolist(),
            "metrics": row,
        }
        write_json(out_dir / f"seq_{idx:04d}_predictions.json", pred_json)

        if args.save_video:
            frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
            vis_frames = []
            for t, frame in enumerate(frames_np):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if cfg["render"]["draw_measurements"]:
                    draw_tracer(bgr, result.measured_uv[: t + 1], color=(255, 255, 0), thickness=1)
                if cfg["render"]["draw_filtered"]:
                    draw_tracer(bgr, result.filtered_uv[: t + 1], color=(0, 255, 255), thickness=cfg["render"]["tracer_thickness"])
                if cfg["render"]["draw_reprojected"]:
                    draw_tracer(bgr, result.uv_reprojected[: t + 1], color=(0, 100, 255), thickness=1)
                vis_frames.append(bgr)
            render_video(vis_frames, out_dir / f"seq_{idx:04d}_overlay.mp4", fps=60.0)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        summary = {k: float(np.mean([r[k] for r in rows])) if k != "sequence" else -1 for k in rows[0].keys()}
        print("Average metrics:")
        for k, v in summary.items():
            if k != "sequence":
                print(f"  {k}: {v:.6f}")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
