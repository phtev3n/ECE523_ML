# Golf Ball Tracer Project

The implementation follows the proposal structure:

1. **Stage 1**: detector/tracker front end that predicts a golf-ball heatmap, sub-pixel offset, confidence, and aleatoric uncertainty.
2. **Stage 2**: Kalman filter state estimation for 2D image-plane smoothing and dropout handling.
3. **Stage 3**: WITB-inspired temporal lifting network that maps 2D tracks plus uncertainty features into 3D trajectory points and end-of-track probability.
4. **Stage 4**: Reprojection and tracer rendering for final video overlay.

## Project layout

```text
golf_tracer_project/
├── configs/
│   ├── detector.yaml
│   ├── trajectory.yaml
│   └── pipeline.yaml
├── golf_tracer/
│   ├── data/
│   │   ├── real_dataset.py
│   │   └── synthetic_dataset.py
│   ├── models/
│   │   ├── detector.py
│   │   ├── losses.py
│   │   └── trajectory_lifter.py
│   ├── tracking/
│   │   ├── kalman.py
│   │   └── pipeline.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── geometry.py
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── render.py
│   │   └── train.py
│   └── __init__.py
├── scripts/
│   ├── train_detector.py
│   ├── train_trajectory.py
│   ├── test_pipeline.py
│   └── make_demo_data.py
├── outputs/
├── requirements.txt
└── README.md
```

## Dataset format for real data

### 2D detector dataset
Store each sequence as:

```text
dataset_root/
├── seq_000/
│   ├── frames/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── annotations.json
```

`annotations.json` example:
```json
{
  "fps": 60.0,
  "camera": {
    "fx": 1400.0,
    "fy": 1400.0,
    "cx": 640.0,
    "cy": 360.0,
    "camera_height_m": 1.2,
    "tilt_deg": -3.0
  },
  "frames": [
    {
      "frame_index": 0,
      "visible": 1,
      "uv": [610.2, 338.7],
      "xyz": [0.0, 0.1, 0.0]
    }
  ]
}
```

### What is used
- `uv`: image-plane ground-truth ball center in pixels
- `visible`: 0 or 1
- `xyz`: optional 3D target in meters for trajectory lifting
- `camera`: used for projection/reprojection utilities

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic demo data
```bash
python scripts/make_demo_data.py --out_dir demo_dataset --num_sequences 200
```

### 3. Train detector
```bash
python scripts/train_detector.py --config configs/detector.yaml --dataset_root demo_dataset
```

### 4. Train 3D lifter
```bash
python scripts/train_trajectory.py --config configs/trajectory.yaml --dataset_root demo_dataset
```

### 5. Test full pipeline + render tracer
```bash
python scripts/test_pipeline.py   --config configs/pipeline.yaml   --dataset_root demo_dataset   --detector_ckpt outputs/detector_best.pt   --trajectory_ckpt outputs/trajectory_best.pt   --save_video
```

## Main outputs
- detector checkpoints
- trajectory checkpoints
- CSV metrics summary
- rendered MP4 tracer overlays
- optional per-frame JSON predictions

## Core metrics
- 2D center RMSE
- visibility F1
- 2D track smoothness
- 3D RMSE
- carry distance error
- apex height error
- landing point error

## Recommended path for your real project
1. Start with the synthetic dataset to validate the pipeline.
2. Replace `SyntheticGolfTrajectoryDataset` with your Trackman-aligned real dataset.
3. Calibrate camera intrinsics/extrinsics for each capture setup.
4. Fine-tune detector on your golf videos.
5. Train the 3D lifter using Trackman-derived 3D/metric targets.
6. Evaluate the three ablations proposed in the PDF:
   - detector only
   - 3D lifter only on GT 2D tracks
   - detector + Kalman + 3D lifter end-to-end pipeline
