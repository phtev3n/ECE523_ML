# Golf Ball Trajectory Estimation — ECE 523 ML Project

**Brian Morgan | Spring 2026**

Single-camera golf ball tracking and 3D trajectory estimation from iPhone video using a synthetic-to-real transfer learning pipeline. The system detects the ball frame-by-frame, smooths the 2D track with a Kalman filter, and lifts it to metric 3D coordinates via an LSTM trajectory network.

---

## Results Summary

| Metric | Value |
|---|---|
| 2D center RMSE (real data) | **25.8 px** |
| Visibility F1 (real data) | **0.92** |
| 3D trajectory RMSE | ~3.0 m *(limited by monocular depth ambiguity and data scarcity)* |
| Real sequences evaluated | 13 (5 PW · 5 7-iron · 3 Driver) |

---

## Pipeline Overview

```
Raw .MOV video
      │
      ▼
detect_impact_frame.py   — audio + motion-based impact detection with GUI confirmation
      │
      ▼
extract_frames.py        — 24-frame clip extracted around impact at 512×512
      │
      ▼
annotate_ball_2d.py      — interactive click-annotation of 2D ball center per frame
      │
      ▼
reconstruct_3d_from_2d.py — monocular ballistic optimizer (16 px mean reprojection error)
      │
      ▼
build_dataset.py         — assembles annotated sequences into real_data_work/dataset/
      │
      ├── train_detector.py (synthetic pretraining)
      │         └── train_detector.py --finetune (real data fine-tune)
      │
      ├── train_trajectory.py — LSTM TrajectoryLifter, optionally from checkpoint
      │
      └── test_pipeline.py   — end-to-end evaluation + tracer overlay video
```

---

## Repository Layout

```text
Project/
├── configs/
│   ├── detector.yaml           — detector training config
│   ├── detector_finetune.yaml  — real-data fine-tune config
│   ├── trajectory.yaml         — LSTM lifter training config
│   └── pipeline.yaml           — end-to-end eval config
│
├── golf_tracer/                — core library
│   ├── data/
│   │   ├── synthetic_dataset.py
│   │   └── real_dataset.py
│   ├── models/
│   │   ├── detector.py         — MultiScaleBallDetector (heatmap + offset + confidence)
│   │   ├── trajectory_lifter.py — LSTM 2D→3D lifter with spin head
│   │   └── losses.py
│   ├── tracking/
│   │   ├── kalman.py           — constant-velocity Kalman filter
│   │   └── pipeline.py        — GolfBallTrackingPipeline (detector → Kalman → lifter)
│   └── utils/
│       ├── config.py · geometry.py · io.py · metrics.py · render.py · train.py
│
├── scripts/
│   ├── detect_impact_frame.py  — GUI tool: find impact frame in raw .MOV
│   ├── extract_frames.py       — extract N frames around impact at target resolution
│   ├── annotate_ball_2d.py     — interactive 2D annotation GUI
│   ├── reconstruct_3d_from_2d.py — single-camera 3D reconstruction from 2D labels
│   ├── build_dataset.py        — assemble annotated shots into dataset
│   ├── make_demo_data.py       — generate synthetic training data
│   ├── train_detector.py       — detector training + fine-tuning
│   ├── train_trajectory.py     — LSTM lifter training
│   ├── test_pipeline.py        — full pipeline eval + overlay video generation
│   ├── compile_demo_reel.py    — concatenate per-shot overlay clips into demo_reel.mp4
│   ├── plot_trajectories.py    — predicted vs GT trajectory plots per sequence
│   ├── plot_club_comparison.py — ideal reference vs pipeline by club category
│   ├── capture_pipeline_stages.py — presentation stage images for one sequence
│   ├── simulate_trajectory.py  — physics ballistic simulator (drag + Magnus lift)
│   ├── estimate_camera_params.py / v2 — camera intrinsic estimation from EXIF
│   └── calibrate_camera.py     — checkerboard geometric calibration
│
├── real_data_work/
│   ├── shots/                  — per-shot annotation files and sequence metadata
│   │   ├── IMG_9737_pw1/ … IMG_9743_pw5/   (5 pitching wedge)
│   │   ├── IMG_9744_7i_1/ … IMG_9748_7i_5/ (5 seven-iron)
│   │   └── IMG_9749_Dr_1/ … IMG_9758_Dr_3/ (3 driver)
│   └── dataset/                — built dataset (seq_0000 – seq_0012)
│       └── seq_XXXX/
│           ├── frames/         — 24 × 512×512 PNG frames
│           └── annotations.json
│
├── real_video_data/
│   └── 60fps/                  — raw iPhone .MOV files (not committed to git)
│
├── outputs/
│   ├── detector_finetune_best.pt
│   ├── trajectory_best.pt
│   ├── real_test_results/      — seq_*_predictions.json
│   ├── demo_videos/            — seq_*_overlay.mp4 + demo_reel.mp4
│   ├── trajectory_plots/       — per-sequence PNG plots + summary grid
│   └── pipeline_stages/        — presentation images from capture_pipeline_stages.py
│
├── future_work/
│   └── limitations_and_future_work.md
│
├── golf_tracer_rebuild_3d.slurm     — HPC: 3D reconstruction + dataset build
├── golf_tracer_demo_video.slurm     — HPC: test pipeline + plots + demo reel
├── requirements.txt
└── README.md
```

---

## Dataset Format

```text
real_data_work/dataset/
└── seq_XXXX/
    ├── frames/
    │   ├── 000000.png   (512×512, frame at impact)
    │   └── 000001.png … 000023.png
    └── annotations.json
```

`annotations.json` schema:
```json
{
  "fps": 60.0,
  "camera": {
    "fx": 1507.6, "fy": 1464.4,
    "cx": 439.4,  "cy": 15.7,
    "camera_height_m": 0.9721
  },
  "frames": [
    { "frame_index": 0, "visible": 1, "uv": [294.0, 390.0], "xyz": [0.0, 0.0, 3.96] }
  ]
}
```

**Camera setup**: single iPhone mounted side-on (perpendicular to target line), lens height 0.972 m, horizontal distance to ball 3.988 m (13 ft 1 in), 60 fps.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic training data
```bash
python scripts/make_demo_data.py --out_dir demo_dataset --num_sequences 500
```

### 3. Train detector on synthetic data
```bash
python scripts/train_detector.py --config configs/detector.yaml --dataset_root demo_dataset
```

### 4. Fine-tune detector on real data
```bash
python scripts/train_detector.py \
    --config configs/detector_finetune.yaml \
    --dataset_root real_data_work/dataset \
    --checkpoint outputs/detector_best.pt
```

### 5. Train trajectory lifter
```bash
python scripts/train_trajectory.py \
    --config configs/trajectory.yaml \
    --dataset_root real_data_work/dataset \
    --checkpoint outputs/trajectory_best.pt
```

### 6. Evaluate full pipeline + render overlay videos
```bash
python scripts/test_pipeline.py \
    --config configs/pipeline.yaml \
    --dataset_root real_data_work/dataset \
    --detector_ckpt outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir outputs/demo_videos
```

### 7. Generate trajectory and club comparison plots
```bash
python scripts/plot_trajectories.py \
    --results_dir outputs/demo_videos \
    --dataset_root real_data_work/dataset \
    --out_dir outputs/trajectory_plots

python scripts/plot_club_comparison.py \
    --results_dir outputs/demo_videos \
    --out_dir outputs/trajectory_plots
```

### 8. Generate presentation pipeline stage images
```bash
python scripts/capture_pipeline_stages.py \
    --seq_dir real_data_work/dataset/seq_0006 \
    --predictions outputs/real_test_results/seq_0006_predictions.json \
    --overlay_mp4 <path>/seq_0006_overlay.mp4 \
    --out_dir outputs/pipeline_stages
```

---

## Sequence-to-Video Mapping

| Sequences | Club | Source videos |
|---|---|---|
| seq_0000 – seq_0004 | Pitching Wedge | IMG_9737 – IMG_9743 |
| seq_0005 – seq_0009 | 7-Iron | IMG_9744 – IMG_9748 |
| seq_0010 – seq_0012 | Driver | IMG_9749, IMG_9757, IMG_9758 |

---

## HPC Jobs (SLURM)

| File | Purpose |
|---|---|
| `golf_tracer_rebuild_3d.slurm` | 3D reconstruction from 2D annotations + dataset assembly |
| `golf_tracer_demo_video.slurm` | Full pipeline eval → overlay videos → trajectory plots |

Submit with `sbatch <file>.slurm` from the project root on the HPC.

---

## Key Design Decisions

- **Synthetic pretraining**: detector trained on procedurally generated ball trajectories with augmented golfer silhouettes, then fine-tuned on 13 annotated real sequences.
- **3D reconstruction**: monocular single-camera ballistic optimizer (`reconstruct_3d_from_2d.py`) using standard pinhole projection (`v = cy + fy*(cam_h - y)/z`). Mean reprojection error: 16 px after correcting projection convention.
- **Magnus lift simulation**: `plot_club_comparison.py` uses a drag + backspin lift model with binary-search Cl tuning to match published amateur carry distances (PW: 120 yd, 7i: 155 yd, Driver: 235 yd).

## Limitations

3D trajectory estimation is fundamentally limited by monocular depth ambiguity and the small real dataset (13 sequences, 24 frames each). See [`future_work/limitations_and_future_work.md`](future_work/limitations_and_future_work.md) for a detailed analysis and improvement roadmap.
