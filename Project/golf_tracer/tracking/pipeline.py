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
        for t in range(T):
            pred = self.detector(frames_tensor[t : t + 1].to(self.device))
            uv_small = decode_heatmap(pred["heatmap"], pred["offset"])[0].cpu().numpy()
            hm_h, hm_w = pred["heatmap"].shape[-2:]
            img_h, img_w = frames_tensor.shape[-2:]
            sx = img_w / hm_w
            sy = img_h / hm_h
            uv = np.array([uv_small[0] * sx, uv_small[1] * sy], dtype=np.float32)
            vis = torch.sigmoid(pred["visible_logit"]).item()
            lv = pred["log_var"][0, :, int(round(uv_small[1])) % hm_h, int(round(uv_small[0])) % hm_w].mean().item()
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

            pred_uv = kf.predict()
            if visible_prob[i] > 0.35:
                filt_uv = kf.update(uv)
            elif kf.can_coast:
                filt_uv = kf.miss()
            else:
                # After extended dropout, re-anchor to the detector measurement.
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
        xyz = traj_pred["xyz"][0].cpu().numpy()
        uv_rep = project_points(xyz, camera)
        return PipelineOutput(
            measured_uv=detector_uv,
            filtered_uv=filtered,
            xyz_pred=xyz,
            uv_reprojected=uv_rep,
            visible_prob=visible_prob,
        )
