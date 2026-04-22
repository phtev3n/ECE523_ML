#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_train_demo
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --time=20:00:00

# --------------------
### Code Section
# --------------------

python --version
pwd

# ── Step 1: Train detector on pre-generated demo_dataset ──
# demo_dataset (500 sequences, 512×512, with golfer silhouettes) is already
# on the HPC from the previous run.  batch_size=16, num_workers=1 per HPC
# recommendation (system max is 1 worker; also reduces disk-read contention).
echo "==> Training detector on demo_dataset (batch=16, workers=1)..."
python scripts/train_detector.py \
    --config       configs/detector.yaml \
    --dataset_root demo_dataset

# ── Step 2: Train trajectory LSTM on pre-generated demo_dataset ──
# load_frames=False is set inside train_trajectory.py — only UV/XYZ/visibility
# are read from annotations.json; no PNG frame files are loaded, so LSTM
# training has negligible memory and IO overhead.
echo "==> Training trajectory LSTM on demo_dataset..."
python scripts/train_trajectory.py \
    --config       configs/trajectory.yaml \
    --dataset_root demo_dataset

# ── Step 3: Smoke-test the full pipeline ──
echo "==> Running pipeline smoke test..."
python scripts/test_pipeline.py \
    --config          configs/pipeline_smoke.yaml \
    --dataset_root    demo_dataset \
    --detector_ckpt   outputs/detector_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/test_results \
    --max_sequences   20

echo "==> Demo training complete. Check outputs/test_results/summary.json"
echo "    Expected: rmse_2d < 5px, visibility_f1 > 0.90 on synthetic test data."
