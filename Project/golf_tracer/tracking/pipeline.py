from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
        kf_init_frame  : Index of the first frame at which the Kalman filter
                         was seeded by a high-confidence detection.  Filtered
                         positions before this index are raw (unsmoothed)
                         detector outputs and should not be drawn as part of
                         the tracer.  -1 if the filter was never initialised
                         (no high-confidence detection in the sequence).
    """
    measured_uv:    np.ndarray
    filtered_uv:    np.ndarray
    xyz_pred:       np.ndarray
    uv_reprojected: np.ndarray
    visible_prob:   np.ndarray
    spin_pred:      np.ndarray   # shape (2,): [backspin_rpm, sidespin_rpm]
    kf_init_frame:  int = -1     # first frame Kalman was seeded; -1 = never


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

    @staticmethod
    def classical_detect(frames_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Classical background-subtraction ball detector for a static camera.

        For a tripod-mounted camera, the golf ball is the only small, fast-moving
        object after impact.  This detector:
          1. Computes a temporal median of all frames as a background estimate.
          2. Per frame: subtracts the background, thresholds, and finds connected
             components (blobs).
          3. Retains only ball-sized blobs (area 5–600 sq px at 512×512, i.e.
             ~2–28 px diameter) — the golfer body and club create much larger blobs.
          4. Among surviving candidates, picks the one closest to the Kalman
             prediction, or the largest small blob when no prior track exists.

        Returns
        -------
        uv : (T, 2) float32 — detected (u, v) in image-pixel space
        vis : (T,) float32  — 1.0 if a ball-sized blob was found, else 0.0
        """
        # Convert to HxWxC uint8 numpy for OpenCV
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        T, H, W = frames_np.shape[:3]

        # --- Background: temporal median (ball is too small/fast to appear) ---
        bg_f = np.median(frames_np.astype(np.float32), axis=0)
        bg_gray = cv2.cvtColor(bg_f.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        uv_out  = np.zeros((T, 2), dtype=np.float32)
        vis_out = np.zeros(T,      dtype=np.float32)

        # Kalman to guide candidate selection across frames
        kf = Kalman2D(dt=1.0 / 60.0, process_var=200.0, meas_var=10.0)

        for t in range(T):
            gray = cv2.cvtColor(frames_np[t], cv2.COLOR_RGB2GRAY).astype(np.float32)
            diff = np.abs(gray - bg_gray)

            # Also add frame-to-frame difference for early frames where the golfer
            # and ball both move (temporal median won't fully remove the golfer)
            if t > 0:
                prev_gray = cv2.cvtColor(frames_np[t - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
                frame_diff = np.abs(gray - prev_gray)
                # Combine: background diff reveals ALL moving objects; frame diff
                # emphasises CURRENTLY moving ones (ball accelerates away faster)
                combined = np.maximum(diff * 0.6, frame_diff * 0.4)
            else:
                combined = diff

            blurred = cv2.GaussianBlur(combined, (5, 5), 1.5)
            _, thresh = cv2.threshold(blurred.astype(np.uint8), 18, 255, cv2.THRESH_BINARY)

            # Morphological open: remove single-pixel noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # --- Blob analysis ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, connectivity=8
            )

            # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA (label 0 = background)
            ball_candidates: list[tuple[float, float, float]] = []  # (cx, cy, area)
            for lbl in range(1, num_labels):
                area = float(stats[lbl, cv2.CC_STAT_AREA])
                # Ball-sized filter: 5–600 sq px.  A golf ball at 5–30 m with
                # a 512-px frame and ~600px focal length subtends ~2–15 px radius
                # → 12–700 sq px area.  Golfer torso: >>1000 sq px.
                if 5.0 <= area <= 600.0:
                    cx, cy = float(centroids[lbl, 0]), float(centroids[lbl, 1])
                    ball_candidates.append((cx, cy, area))

            if ball_candidates:
                if kf.initialized:
                    # Kalman predict gives expected position; pick nearest candidate
                    pred_uv = kf.predict()
                    best = min(ball_candidates,
                               key=lambda c: (c[0] - pred_uv[0]) ** 2 + (c[1] - pred_uv[1]) ** 2)
                    kf.update([best[0], best[1]])
                else:
                    # First detection: pick the largest small blob
                    best = max(ball_candidates, key=lambda c: c[2])
                    kf.init([best[0], best[1]])

                uv_out[t]  = [best[0], best[1]]
                vis_out[t] = 1.0
            else:
                if kf.initialized and kf.can_coast:
                    pred_uv = kf.predict()
                    kf.miss()
                    uv_out[t] = pred_uv
                else:
                    uv_out[t] = uv_out[t - 1] if t > 0 else np.array([W / 2, H / 2])

        return uv_out, vis_out

    @torch.no_grad()
    def run_sequence(
        self,
        frames_tensor: torch.Tensor,
        camera: dict,
        precomputed_uv: np.ndarray | None = None,
        precomputed_vis: np.ndarray | None = None,
        use_classical: bool = False,
    ) -> PipelineOutput:
        """Run the full pipeline on a pre-loaded frame sequence.

        Args:
            frames_tensor : (T, 3, H, W) normalised RGB frames
            camera        : dict with pinhole intrinsics (fx, fy, cx, cy,
                            camera_height_m) matching the annotations.json format

        Returns:
            PipelineOutput dataclass (see field docstrings above)
        """
        T = frames_tensor.shape[0]
        img_h, img_w = frames_tensor.shape[-2:]

        vis_threshold            = float(self.config.get("tracking", {}).get("visibility_threshold", 0.35))
        force_measurement_update = bool(self.config.get("tracking", {}).get("force_measurement_update", False))
        use_motion_gate          = self.config.get("tracking", {}).get("motion_gate", True)

        detector_uv  = []
        visible_prob = []
        uncertainty  = []

        # ---- Stage 1: Detection ----
        # Three modes:
        #   precomputed_uv : ground-truth UV passed in directly (diagnostic / ceiling test)
        #   use_classical  : temporal background subtraction + blob size filter
        #   default        : learned ResNet50 detector (with optional motion gate)

        if precomputed_uv is not None:
            # Ground-truth bypass — skip both detection AND Kalman.
            # The GT positions are exact; filtering would only add error.
            uv_arr  = precomputed_uv.astype(np.float32)
            vis_arr = precomputed_vis.astype(np.float32) if precomputed_vis is not None else np.ones(T, dtype=np.float32)
            detector_uv  = uv_arr
            visible_prob = vis_arr
            uncertainty  = np.zeros(T, dtype=np.float32)
            filtered = uv_arr.copy()   # bypass Kalman entirely

            # Flip u,v to match synthetic training convention.
            # Real frames were extracted with --orient 180 so annotations have
            # v INCREASING as the ball rises; synthetic uses v = fy*(h-y)/z + cy
            # which has v DECREASING as the ball rises.  Flip both axes so the
            # LSTM sees the same coordinate convention it was trained on.
            lstm_uv = filtered.copy()
            lstm_uv[:, 0] = (img_w - 1) - lstm_uv[:, 0]
            lstm_uv[:, 1] = (img_h - 1) - lstm_uv[:, 1]

            traj_features = np.concatenate([
                lstm_uv,
                visible_prob[:, None],
                uncertainty[:, None],
                np.arange(T, dtype=np.float32)[:, None] / max(T - 1, 1),
            ], axis=1)
            traj_tensor = torch.from_numpy(traj_features[None]).float().to(self.device)
            traj_pred   = self.trajectory_model(traj_tensor)
            xyz  = traj_pred["xyz"][0].detach().cpu().numpy()
            spin = traj_pred["spin"][0].detach().cpu().numpy() if "spin" in traj_pred else np.zeros(2, dtype=np.float32)
            uv_rep = project_points(xyz, camera)
            return PipelineOutput(
                measured_uv=uv_arr, filtered_uv=filtered,
                xyz_pred=xyz, uv_reprojected=uv_rep,
                visible_prob=vis_arr, spin_pred=spin,
                kf_init_frame=0,   # GT bypass: all frames valid from the start
            )

        elif use_classical:
            # Classical motion detector — no domain gap, no GPU required
            uv_cl, vis_cl = self.classical_detect(frames_tensor)
            detector_uv  = list(uv_cl)
            visible_prob = list(vis_cl)
            uncertainty  = [0.0] * T

        else:
            # ---- Learned detector (per-frame) with optional motion gate ----
            for t in range(T):
                frame = frames_tensor[t : t + 1].to(self.device)
                pred  = self.detector(frame)

                hm_h, hm_w = pred["heatmap"].shape[-2:]

                if use_motion_gate and t > 0:
                    prev = frames_tensor[t - 1 : t].to(self.device)
                    diff = (frame - prev).abs().mean(dim=1, keepdim=True)

                    # ---- Stage A: high-pass filter (frequency gate) ----
                    # Subtracting a blurred copy leaves only fine-scale detail.
                    # The golfer body creates a large low-frequency blob in the
                    # diff image; the ball creates a sharp high-frequency spike.
                    blur_sigma = float(self.config.get("tracking", {}).get(
                        "motion_gate_blur_sigma", 8.0
                    ))
                    k = max(3, int(blur_sigma * 2) | 1)   # odd kernel
                    pad = k // 2
                    diff_blur = F.avg_pool2d(
                        F.pad(diff, [pad, pad, pad, pad], mode="reflect"),
                        kernel_size=k, stride=1,
                    )
                    diff_hp = (diff - diff_blur).clamp(min=0.0)

                    # ---- Stage B: blob-size gate (area filter) ----
                    # After thresholding the high-pass diff, suppress contiguous
                    # regions larger than a ball-sized area.  The golf club is
                    # thin but elongated — it passes the frequency gate but has
                    # a much larger bounding box than the ball.  We dilate and
                    # erode to find large blobs, then subtract them from the gate.
                    # Implemented via max-pool (dilation proxy) followed by
                    # average-pool (area estimator): regions that survive are
                    # spatially compact (ball-sized).
                    # Ball diameter at 512px: ~4-20px → at heatmap 1/4 scale: 1-5px
                    # Club / body blob: >>20px at heatmap scale
                    gate_cfg = self.config.get("tracking", {})
                    ball_max_px = int(gate_cfg.get("motion_gate_ball_max_px", 12))

                    diff_small_raw = F.interpolate(
                        diff_hp, size=(hm_h, hm_w), mode="bilinear", align_corners=False
                    )
                    # Detect large blobs: dilate with max-pool, then check if
                    # average neighbourhood is also high (large connected region).
                    large_k = max(3, ball_max_px | 1)
                    large_pad = large_k // 2
                    # Max-pool spreads any hot pixel over a ball_max_px window
                    large_blob = F.max_pool2d(
                        F.pad(diff_small_raw, [large_pad, large_pad, large_pad, large_pad], mode="reflect"),
                        kernel_size=large_k, stride=1,
                    )
                    # Average within the same window: high avg + high max = large blob
                    large_avg = F.avg_pool2d(
                        F.pad(large_blob, [large_pad, large_pad, large_pad, large_pad], mode="reflect"),
                        kernel_size=large_k, stride=1,
                    )
                    # Suppress pixels that belong to large blobs (body/club):
                    # where both local max and local average are high, that's a
                    # large region — zero it out.  Small compact objects (ball)
                    # have high local max but low local average.
                    size_mask = (large_avg < 0.3 * large_blob + 1e-6).float()
                    diff_size_gated = diff_small_raw * size_mask

                    diff_norm = diff_size_gated / (diff_size_gated.amax(dim=(-2, -1), keepdim=True) + 1e-6)
                    gated_heatmap = pred["heatmap"] * (0.2 + 0.8 * diff_norm)
                else:
                    gated_heatmap = pred["heatmap"]

                uv_small = decode_heatmap(gated_heatmap, pred["offset"])[0].detach().cpu().numpy()

                sx = img_w / hm_w
                sy = img_h / hm_h
                uv = np.array([uv_small[0] * sx, uv_small[1] * sy], dtype=np.float32)
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

        detector_uv  = np.asarray(detector_uv,  dtype=np.float32)
        visible_prob = np.asarray(visible_prob, dtype=np.float32)
        uncertainty  = np.asarray(uncertainty,  dtype=np.float32)

        # ---- Stage 2: Kalman filtering ----
        kf = Kalman2D(**self.config["kalman"])
        filtered = []
        kf_init_frame = -1   # first frame where Kalman was seeded

        # Minimum confidence required to seed the filter.  Initialising on the
        # first frame unconditionally causes divergence when the first detection
        # is a false positive (e.g. detector locks onto a static background
        # region before the ball enters the frame).  Waiting for a genuinely
        # high-confidence detection prevents the filter from being seeded at
        # the wrong position and coasting away from the true ball trajectory.
        vis_init_threshold = float(
            self.config.get("tracking", {}).get("vis_init_threshold", 0.5)
        )

        for i, uv in enumerate(detector_uv):
            if not kf.initialized:
                if visible_prob[i] >= vis_init_threshold:
                    kf.init(uv)
                    kf_init_frame = i
                # Output the raw detection regardless — the filtered track
                # will match the measurement until the filter is seeded, which
                # is better than outputting a stale/wrong initialisation.
                filtered.append(uv.copy())
                continue

            kf.predict()

            if force_measurement_update:
                # Always update regardless of visibility — used for debugging only
                filt_uv = kf.update(uv)
            else:
                if visible_prob[i] > vis_threshold and kf.gate(uv):
                    # High-confidence detection that passes the Mahalanobis gate:
                    # incorporate the measurement into the state estimate.
                    filt_uv = kf.update(uv)
                elif visible_prob[i] > vis_threshold and not kf.gate(uv):
                    # Detector is confident but the position is geometrically
                    # inconsistent with the current trajectory (e.g. detector
                    # jumped to the golfer body).  Treat as a missed detection.
                    filt_uv = kf.miss()
                elif kf.can_coast:
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
        #
        # Flip u,v to match synthetic training convention.
        # Real frames were extracted with --orient 180 so annotations have
        # v INCREASING as the ball rises; synthetic uses v = fy*(h-y)/z + cy
        # which has v DECREASING as the ball rises.  Flip both axes so the
        # LSTM sees the same coordinate convention it was trained on.
        lstm_uv = filtered.copy()
        lstm_uv[:, 0] = (img_w - 1) - lstm_uv[:, 0]
        lstm_uv[:, 1] = (img_h - 1) - lstm_uv[:, 1]

        traj_features = np.concatenate(
            [
                lstm_uv,                                                         # (T, 2) flipped
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

        # Apply physical plausibility constraints to the LSTM output.
        # The below_ground_loss penalises y < 0 during training but does not
        # guarantee it at inference (especially on out-of-distribution inputs).
        # Clamping here prevents downstream metric extrapolation (carry, apex,
        # ToF) from producing nonsensical results when the model mispredicts.
        #   y (height above ground) : must be >= 0
        #   z (depth, camera-forward): must be positive (ball in front of camera)
        xyz[:, 1] = np.clip(xyz[:, 1], 0.0, None)
        xyz[:, 2] = np.clip(xyz[:, 2], 0.1, None)

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
            kf_init_frame=kf_init_frame,
        )
