from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from golf_tracer.models.detector import decode_heatmap
from golf_tracer.tracking.kalman import Kalman2D
from golf_tracer.utils.geometry import project_points


@dataclass
class PipelineOutput:
    """All outputs produced by a single run of GolfBallTrackingPipeline.

    Attributes:
        measured_uv    : (T, 2) Raw detector detections in image-pixel space.
                         These are noisy and may be invalid on occluded frames.
        filtered_uv    : (T, 2) Kalman-filtered positions.  Smoother than
                         measured_uv; used as input features for the LSTM.
        xyz_pred       : (T, 3) 3D positions predicted by TrajectoryLifter,
                         in camera-frame metres (x=lateral, y=height, z=depth).
        uv_reprojected : (T, 2) xyz_pred reprojected into image space using
                         the pinhole camera model.  Visualised as the orange
                         tracer in the overlay video.
        visible_prob   : (T,)  Per-frame visibility probability output by the
                         detector's visibility head after sigmoid.
        spin_pred      : (2,)  Sequence-level spin estimate [backspin_rpm,
                         sidespin_rpm] from the LSTM spin_head.  Near-zero
                         values indicate the model lacked sufficient clip
                         length to observe Magnus curvature; the physics
                         fitting fallback should be used instead.
    """
    measured_uv:    np.ndarray
    filtered_uv:    np.ndarray
    xyz_pred:       np.ndarray
    uv_reprojected: np.ndarray
    visible_prob:   np.ndarray
    spin_pred:      np.ndarray   # shape (2,): [backspin_rpm, sidespin_rpm]


class GolfBallTrackingPipeline:
    """End-to-end inference pipeline: frames → 3D trajectory + ball metrics.

    Processing flow per sequence
    ----------------------------
    1. Detector stage (per frame):
         Each frame is passed through MultiScaleBallDetector to obtain:
           - heatmap + offset  → decoded to raw (u, v) detection
           - visible_logit     → sigmoid → visibility probability
           - log_var           → aleatoric uncertainty of the detection

    2. Kalman filtering (per frame):
         A 2D constant-velocity Kalman filter smooths the detector outputs.
         On high-confidence frames (visible_prob > threshold) the filter is
         updated with the measured position.  On low-confidence frames it
         either coasts (propagates without update) for up to max_coast_frames
         or re-initialises if coasting budget is exhausted.

    3. Feature assembly (sequence):
         Filtered positions, visibility probabilities, uncertainty, and a
         normalised time index are concatenated into a (T, 5) feature tensor.

    4. TrajectoryLifter (sequence):
         The LSTM processes the feature sequence and outputs:
           - xyz_pred  : per-frame 3D positions (metres, camera-frame)
           - eot_prob  : per-frame landing probability
           - spin      : sequence-level backspin / sidespin estimate (rpm)

    5. Reprojection:
         xyz_pred is projected back into image space using the pinhole camera
         model for visualisation as the 3D-consistent tracer overlay.
    """

    def __init__(self, detector, trajectory_model, config: dict, device: torch.device):
        self.detector         = detector
        self.trajectory_model = trajectory_model
        self.config           = config
        self.device           = device

    @torch.no_grad()
    def run_sequence(self, frames_tensor: torch.Tensor, camera: dict) -> PipelineOutput:
        """Run the full pipeline on a pre-loaded frame sequence.

        Args:
            frames_tensor : (T, 3, H, W) normalised RGB frames
            camera        : dict with pinhole intrinsics (fx, fy, cx, cy,
                            camera_height_m) matching the annotations.json format

        Returns:
            PipelineOutput dataclass (see field docstrings above)
        """
        T = frames_tensor.shape[0]

        detector_uv  = []
        visible_prob = []
        uncertainty  = []

        vis_threshold          = float(self.config.get("tracking", {}).get("visibility_threshold", 0.35))
        force_measurement_update = bool(self.config.get("tracking", {}).get("force_measurement_update", False))

        img_h, img_w = frames_tensor.shape[-2:]

        # ---- Stage 1: Per-frame detection ----
        for t in range(T):
            frame = frames_tensor[t : t + 1].to(self.device)
            pred  = self.detector(frame)

            # Decode peak heatmap cell + sub-pixel offset → heatmap-space (u, v)
            uv_small = decode_heatmap(pred["heatmap"], pred["offset"])[0].detach().cpu().numpy()
            hm_h, hm_w = pred["heatmap"].shape[-2:]

            # Scale from heatmap resolution back to full image resolution
            sx = img_w / hm_w
            sy = img_h / hm_h
            uv = np.array([uv_small[0] * sx, uv_small[1] * sy], dtype=np.float32)
            uv[0] = np.clip(uv[0], 0.0, img_w - 1.0)
            uv[1] = np.clip(uv[1], 0.0, img_h - 1.0)

            vis = torch.sigmoid(pred["visible_logit"]).view(-1)[0].item()

            # Read uncertainty from the peak heatmap cell (mean over x/y dims)
            if "log_var" in pred:
                x_idx = int(np.clip(round(float(uv_small[0])), 0, hm_w - 1))
                y_idx = int(np.clip(round(float(uv_small[1])), 0, hm_h - 1))
                lv = pred["log_var"][0, :, y_idx, x_idx].mean().item()
            else:
                lv = 0.0

            detector_uv.append(uv)
            visible_prob.append(vis)
            uncertainty.append(lv)

        detector_uv  = np.asarray(detector_uv,  dtype=np.float32)
        visible_prob = np.asarray(visible_prob, dtype=np.float32)
        uncertainty  = np.asarray(uncertainty,  dtype=np.float32)

        # ---- Stage 2: Kalman filtering ----
        kf = Kalman2D(**self.config["kalman"])
        filtered = []

        for i, uv in enumerate(detector_uv):
            if not kf.initialized:
                kf.init(uv)
                filtered.append(uv.copy())
                continue

            kf.predict()

            if force_measurement_update:
                # Always update regardless of visibility — used for debugging only
                filt_uv = kf.update(uv)
            else:
                if visible_prob[i] > vis_threshold:
                    # High-confidence detection: incorporate the measurement
                    filt_uv = kf.update(uv)
                elif getattr(kf, "can_coast", False):
                    # Low confidence but coasting budget remains: propagate only
                    filt_uv = kf.miss()
                else:
                    # Coasting budget exhausted: re-initialise at current detection
                    kf.init(uv)
                    filt_uv = uv.copy()

            filtered.append(filt_uv)

        filtered = np.asarray(filtered, dtype=np.float32)

        # ---- Stage 3: Feature assembly (T, 5) ----
        # Columns: [u, v, vis_prob, log_var, normalised_t]
        # The normalised time index lets the LSTM infer trajectory phase
        # (launch / apex / descent) without needing explicit fps information.
        traj_features = np.concatenate(
            [
                filtered,                                                        # (T, 2)
                visible_prob[:, None],                                           # (T, 1)
                uncertainty[:, None],                                            # (T, 1)
                np.arange(T, dtype=np.float32)[:, None] / max(T - 1, 1),       # (T, 1)
            ],
            axis=1,
        )

        # ---- Stage 4: TrajectoryLifter ----
        traj_tensor = torch.from_numpy(traj_features[None]).float().to(self.device)
        traj_pred   = self.trajectory_model(traj_tensor)
        xyz = traj_pred["xyz"][0].detach().cpu().numpy()   # (T, 3)

        # Extract spin from spin_head; fall back to zeros for old checkpoints
        # that predate the spin_head addition (they won't have "spin" in output)
        if "spin" in traj_pred:
            spin = traj_pred["spin"][0].detach().cpu().numpy()
        else:
            spin = np.zeros(2, dtype=np.float32)

        # ---- Stage 5: Reprojection for overlay visualisation ----
        uv_rep = project_points(xyz, camera)

        return PipelineOutput(
            measured_uv=detector_uv,
            filtered_uv=filtered,
            xyz_pred=xyz,
            uv_reprojected=uv_rep,
            visible_prob=visible_prob,
            spin_pred=spin,
        )
