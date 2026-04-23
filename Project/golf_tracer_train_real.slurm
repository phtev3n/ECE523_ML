#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_train_real
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

# ── Step 0: Propagate corrected camera intrinsics to all seq_XXXX/annotations.json ──
# The original camera_params.json had cy=15.7 (bad checkerboard calibration).
# camera_params.json has been corrected to cy=256 (centered, EXIF-derived).
# This propagates the fix to the dataset without rebuilding from scratch.
echo "==> Fixing camera intrinsics in dataset..."
python scripts/fix_camera_intrinsics.py \
    --camera_params real_data_work/camera_params.json \
    --dataset_dir   real_data_work/dataset

# ── Step 1: Fine-tune detector on real data (mixed synthetic+real) ──
# Starts from outputs/detector_best.pt produced by golf_tracer_train_demo.slurm
# (golfer-aware synthetic training).  No need to re-run synthetic training here.
echo "==> Fine-tuning detector on real data..."
python scripts/train_detector.py \
    --config       configs/detector_finetune.yaml \
    --dataset_root real_data_work/dataset \
    --checkpoint   outputs/detector_best.pt \
    --save_path    outputs/detector_finetune.pt \
    --mix_synthetic

# ── Step 2: Evaluate corrected pipeline on real data (3 test modes) ──
# trajectory_best.pt from golf_tracer_train_demo.slurm is reused directly.
# Pass 1 — GT UV ceiling: if rmse_3d is still high, the LSTM or intrinsics need work.
echo "==> Testing: GT UV bypass (LSTM ceiling)..."
python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/real_test_results_gt_uv \
    --use_gt_uv

# Pass 2 — Classical background-subtraction baseline (no domain gap).
echo "==> Testing: classical background-subtraction detector..."
python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/real_test_results_classical \
    --classical_detector

# Pass 3 — Full learned pipeline with fine-tuned golfer-aware detector.
echo "==> Testing: full learned pipeline (golfer-aware detector + corrected intrinsics)..."
python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/real_test_results

echo "==> All steps complete."
echo "==> Check outputs/real_test_results/summary.json for rmse_2d and visible_prob_mean."
echo "==> If rmse_2d < 50px and visible_prob_mean < 0.95, proceed to 3D reconstruction fix."
echo "==> To fix 3D labels: run golf_tracer_rebuild_3d.slurm"
