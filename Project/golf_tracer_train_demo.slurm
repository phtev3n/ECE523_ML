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
#SBATCH --time=08:00:00

# --------------------
### Code Section
# --------------------

python --version
pwd

# ── Step 1: Regenerate demo dataset ──
# IMPORTANT: The old demo_dataset was at 256×256 with no golfer silhouettes.
# Fixes applied:
#   - image_size 512×512 matches detector.yaml and real pipeline resolution.
#     RealGolfSequenceDataset loads frames at stored resolution with no resize,
#     so a 256×256 dataset fed into a 512×512 model was a silent size mismatch.
#   - synthetic_dataset.make_background() now renders a golfer silhouette
#     (torso/arms/legs/shoes/club) in 70% of frames so the detector learns
#     to distinguish the ball from the human figure.
echo "==> Regenerating demo dataset at 512x512 with golfer silhouettes..."
python scripts/make_demo_data.py \
    --out_dir        demo_dataset \
    --num_sequences  500 \
    --sequence_length 24 \
    --image_size     512 512

# ── Step 2: Train detector on pre-generated demo dataset ──
# Uses detector.yaml (image_size 512x512, sigma=2.0, fixed heatmap BCE loss).
echo "==> Training detector on demo dataset..."
python scripts/train_detector.py \
    --config       configs/detector.yaml \
    --dataset_root demo_dataset

# ── Step 3: Train trajectory LSTM on pre-generated demo dataset ──
echo "==> Training trajectory LSTM on demo dataset..."
python scripts/train_trajectory.py \
    --config       configs/trajectory.yaml \
    --dataset_root demo_dataset

# ── Step 4: Smoke-test the full pipeline on demo data ──
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
