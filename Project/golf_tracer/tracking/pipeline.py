from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from golf_tracer.models.detector import decode_heatmap
from golf_tracer.tracking.kalman import Kalman2D
from golf_tracer.utils.geometry import project_points


@dataclass
class PipelineOutput:
    measured_uv: np.ndarray
    filtered_uv: np.ndarray
    xyz_pred: np.ndarray
    uv_reprojected: np.ndarray
    visible_prob: np.ndarray
    spin_pred: np.ndarray   # shape (2,): [backspin_rpm, sidespin_rpm] from spin_head


class GolfBallTrackingPipeline:
    def __init__(self, detector, trajectory_model, config: dict, device: torch.device):
        self.detector = detector
        self.trajectory_model = trajectory_model
        self.config = config
        self.device = device

    @torch.no_grad()
    def run_sequence(self, frames_tensor: torch.Tensor, camera: dict) -> PipelineOutput:
        T = frames_tensor.shape[0]

        detector_uv = []
        visible_prob = []
        uncertainty = []

        vis_threshold = float(self.config.get("tracking", {}).get("visibility_threshold", 0.35))
        force_measurement_update = bool(self.config.get("tracking", {}).get("force_measurement_update", False))

        img_h, img_w = frames_tensor.shape[-2:]

        for t in range(T):
            frame = frames_tensor[t : t + 1].to(self.device)
            pred = self.detector(frame)

            uv_small = decode_heatmap(pred["heatmap"], pred["offset"])[0].detach().cpu().numpy()
            hm_h, hm_w = pred["heatmap"].shape[-2:]

            sx = img_w / hm_w
            sy = img_h / hm_h

            uv = np.array(
                [
                    uv_small[0] * sx,
                    uv_small[1] * sy,
                ],
                dtype=np.float32,
            )

            # Clamp decoded image-plane coordinates
            uv[0] = np.clip(uv[0], 0.0, img_w - 1.0)
            uv[1] = np.clip(uv[1], 0.0, img_h - 1.0)

            vis = torch.sigmoid(pred["visible_logit"]).view(-1)[0].item()

            if "log_var" in pred:
                x_idx = int(np.clip(round(float(uv_small[0])), 0, hm_w - 1))
                y_idx = int(np.clip(round(float(uv_small[1])), 0, hm_h - 1))
                lv = pred["log_var"][0, :, y_idx, x_idx].mean().item()
            else:
                lv = 0.0

            detector_uv.append(uv)
            visible_prob.append(vis)
            uncertainty.append(lv)

        detector_uv = np.asarray(detector_uv, dtype=np.float32)
        visible_prob = np.asarray(visible_prob, dtype=np.float32)
        uncertainty = np.asarray(uncertainty, dtype=np.float32)

        kf = Kalman2D(**self.config["kalman"])
        filtered = []

        for i, uv in enumerate(detector_uv):
            if not kf.initialized:
                kf.init(uv)
                filtered.append(uv.copy())
                continue

            kf.predict()

            if force_measurement_update:
                filt_uv = kf.update(uv)
            else:
                if visible_prob[i] > vis_threshold:
                    filt_uv = kf.update(uv)
                elif getattr(kf, "can_coast", False):
                    filt_uv = kf.miss()
                else:
                    kf.init(uv)
                    filt_uv = uv.copy()

            filtered.append(filt_uv)

        filtered = np.asarray(filtered, dtype=np.float32)

        traj_features = np.concatenate(
            [
                filtered,
                visible_prob[:, None],
                uncertainty[:, None],
                np.arange(T, dtype=np.float32)[:, None] / max(T - 1, 1),
            ],
            axis=1,
        )

        traj_tensor = torch.from_numpy(traj_features[None]).float().to(self.device)
        traj_pred = self.trajectory_model(traj_tensor)
        xyz = traj_pred["xyz"][0].detach().cpu().numpy()
        uv_rep = project_points(xyz, camera)

        if "spin" in traj_pred:
            spin = traj_pred["spin"][0].detach().cpu().numpy()
        else:
            spin = np.zeros(2, dtype=np.float32)

        return PipelineOutput(
            measured_uv=detector_uv,
            filtered_uv=filtered,
            xyz_pred=xyz,
            uv_reprojected=uv_rep,
            visible_prob=visible_prob,
            spin_pred=spin,
        )