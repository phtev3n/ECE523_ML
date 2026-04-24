# Current Limitations and Future Work

**ECE 523 — Machine Learning | Golf Ball Trajectory Estimation**
Brian Morgan | Spring 2026

---

## Overview

The current pipeline achieves strong two-dimensional ball detection performance (RMSE₂D = 25.8 px, Visibility F1 = 0.92) on real iPhone golf swing videos, demonstrating that the synthetic-to-real transfer learning approach is viable. However, three-dimensional trajectory estimation remains a significant open challenge, with the LSTM TrajectoryLifter producing physically implausible carry distances and launch metrics when evaluated against ground-truth club data. This document identifies the root causes of those limitations and outlines a concrete path to production-quality 3D trajectory estimation.

---

## 1. Limitations of the Current System

### 1.1 Insufficient Real Training Data for 3D Lifting

The LSTM TrajectoryLifter was fine-tuned on 13 annotated real sequences — approximately 4 sequences per club type (pitching wedge, 7-iron, driver). This is insufficient for the model to learn the relationship between two-dimensional pixel motion and three-dimensional physical velocity across the full trajectory envelope of each club. The consequence is that the extrapolated carry and launch metrics are physically implausible: the model predicts driver-level carry distances for pitching wedge shots and vice versa. A minimum of 150 sequences (50 per club category) is estimated to be necessary for the model to begin generalizing meaningfully within club types, and approximately 500 sequences would be required for robust production-quality estimation.

### 1.2 Short Clip Length Limits Velocity Estimation

Each sequence currently consists of 24 frames extracted from the point of impact, corresponding to 0.4 seconds of ball flight at 60 fps. This captures only the initial portion of the trajectory — before the ball reaches apex — which provides insufficient signal for the LSTM to reliably estimate launch velocity and carry distance. A pitching wedge shot has a total flight time of approximately 2–3 seconds (120–180 frames at 60 fps); a driver shot requires 4–6 seconds (240–360 frames). Truncating the observation to 0.4 seconds forces the model to extrapolate the remaining 85–93% of the flight from the initial impulse alone, which is inherently underdetermined without strong physical priors.

### 1.3 Noisy 3D Ground-Truth Labels

In the absence of a launch monitor, three-dimensional ground-truth trajectories are reconstructed from two-dimensional manual annotations using a single-camera ballistic optimizer (`reconstruct_3d_from_2d.py`). After correcting the camera projection convention and intrinsic parameters, the mean reprojection error was reduced from 116 px to 16 px, which represents a meaningful improvement. However, a 16 px residual at 512-pixel image resolution still introduces substantial label noise into LSTM training — particularly in the depth axis (Z), which is the least constrained dimension in monocular projection. The estimated LSTM 3D ceiling under these label conditions is approximately 3.0–3.4 m RMSE, consistent with observed results.

### 1.4 Monocular Depth Ambiguity

The camera configuration is a single fixed iPhone mounted perpendicular to the target line. This single-view geometry provides no direct depth constraint — depth must be inferred entirely from the apparent size and velocity of the ball across frames, which is strongly confounded by ball distance from the camera. Without a stereo rig, a second camera at an independent angle, or launch monitor data, this ambiguity is a fundamental limitation that cannot be resolved through increased training data alone.

### 1.5 Residual Domain Gap in Ball Detection

While the 2D detector performs well overall, it was trained on synthetic data augmented with procedurally generated golfer silhouettes. The synthetic golfer rendering is a geometric approximation — a collection of colored rectangles and ellipses — and does not capture the full visual complexity of real golfers under varying lighting, clothing, and motion blur conditions. Sequences where the ball passes close to the golfer's body or club head during impact remain the most error-prone for the detector, as evidenced by the higher per-sequence RMSE in those frames.

---

## 2. Recommendations for Future Work

### 2.1 Expand the Annotated Real-Data Corpus

The single highest-impact improvement is expanding the annotated dataset to at least 150 sequences, with a minimum of 50 per club category. Given the annotation throughput of the current interactive tool (approximately 2–3 minutes per sequence), a corpus of 150 sequences represents roughly 5–8 hours of annotation effort — a realistic scope for a focused data collection session.

When structuring the collection, shot variety within each club type matters as much as total count. The dataset should include a representative mix of shot shapes (draw, straight, fade), contact quality (solid, thin, heavy), and trajectory height. A model trained only on "textbook" swings will fail on the natural variation present in real rounds.

### 2.2 Extend Clip Length to Full Ball Flight

The clip extraction pipeline should be modified to capture ball flight from impact until the ball lands or exits the frame, rather than truncating at 24 frames. This single change eliminates the most significant source of velocity estimation error in the current system. The `extract_frames.py` script supports an arbitrary `--num_frames` argument; the recommended setting is 300 frames (5 seconds at 60 fps), which covers the full flight of all club types. The LSTM architecture and training pipeline are already designed to handle variable-length sequences.

### 2.3 Integrate Launch Monitor Ground Truth

The most direct path to high-fidelity 3D ground-truth labels is integration with a consumer launch monitor. Devices such as the Garmin Approach R10, Rapsodo MLM2PRO, or SkyTrak+ measure ball speed, launch angle, launch direction, and spin rate at impact with sub-1% error under controlled conditions. This data, combined with a standard ballistic simulation (already implemented in `scripts/simulate_trajectory.py`), produces physically consistent XYZ trajectories without any dependence on monocular video reconstruction. The pipeline already includes a Trackman data parser (`scripts/parse_trackman.py`) and a `build_dataset.py` mode that accepts externally provided trajectories — launch monitor integration is an incremental engineering task, not an architectural change.

### 2.4 Add a Second Camera Angle

A single camera perpendicular to the target line (side-on view) provides strong constraints on height Y and downrange distance Z but is nearly blind to lateral deviation X. Adding a second camera at approximately 45° to the target line, synchronized with the primary camera, allows triangulation of all three spatial coordinates without dependence on a launch monitor or ballistic model. Standard stereo reconstruction methods can then provide dense, accurate 3D supervision at each frame. This approach requires modest additional hardware (a second tripod-mounted iPhone) and a synchronization mechanism (a clap board or audio sync), but would fundamentally resolve the monocular depth ambiguity identified in Section 1.4.

### 2.5 Improve Synthetic Golfer Rendering

To further close the synthetic-to-real domain gap in the 2D detector, the procedural golfer silhouette renderer should be replaced or augmented with one of the following approaches:

- **Pose-conditioned human rendering**: Use a parametric body model (e.g., SMPL) or a generative model conditioned on golf swing pose sequences to produce photorealistic golfer appearances across the full swing cycle.
- **Real background compositing**: Extract real golfer silhouettes from a small set of annotated frames and composite them onto synthetic ball-trajectory backgrounds. This requires only a handful of real golfer images to substantially improve detection robustness near the body.
- **Targeted hard-negative mining**: Identify the specific frames where the current detector most frequently produces false positives (typically impact and early follow-through) and over-sample those frames during fine-tuning.

### 2.6 Increase Sequence Length Consistency and Camera Calibration Rigor

The current camera intrinsic parameters were derived from EXIF metadata and a centered-principal-point assumption rather than a formal geometric calibration. A one-time multi-image checkerboard calibration with the specific iPhone used for data collection — at the specific video resolution and frame rate used in production — would reduce the reprojection residual below its current 16 px floor and improve 3D reconstruction accuracy for all subsequently collected sequences. This calibration needs to be performed only once and the resulting parameters stored in `camera_params.json`.

---

## 3. Summary

The following table summarizes the estimated impact of each proposed improvement relative to the current system:

| Improvement | Primary Metric Affected | Estimated Effort |
|---|---|---|
| Expand to 150+ annotated sequences | RMSE 3D, carry/launch accuracy | 5–8 hrs annotation |
| Extend clip length to full flight | RMSE 3D, velocity estimation | 1–2 hrs engineering |
| Launch monitor ground truth | RMSE 3D (ceiling ~0.5 m) | Hardware + 2 hrs integration |
| Second camera angle | RMSE 3D, lateral accuracy | Hardware + 4 hrs engineering |
| Improved synthetic golfer rendering | RMSE 2D near-impact | 8–16 hrs engineering |
| Formal camera calibration | Reprojection residual, RMSE 3D | 1–2 hrs data collection |

The two-dimensional ball detection component of the pipeline is production-ready for controlled single-camera setups. Future development effort should be concentrated on the 3D estimation pathway, where the combination of expanded real data, full-flight clips, and launch monitor integration represents a clear and achievable path to physically meaningful trajectory reconstruction.
