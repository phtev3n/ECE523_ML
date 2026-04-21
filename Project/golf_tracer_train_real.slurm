#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_train_real
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --time=06:00:00

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

# ── Step 1: Train detector from scratch on synthetic data (golfer distractors added) ──
# Uses synthetic_dataset.py with _draw_golfer() — the model now sees golfers
# in 70% of training frames and must learn to suppress them.
# Fixed: heatmap BCE double-sigmoid bug, sigma reduced 5.0 → 2.0.
echo "==> Training detector on synthetic data (golfer-aware)..."
python scripts/train_detector.py \
    --config configs/detector.yaml

# ── Step 2: Fine-tune detector on real data (mixed synthetic+real) ──
# Starting from the synthetic checkpoint so the model retains golfer-suppression
# knowledge while adapting to real appearance (grass texture, lighting, actual ball).
echo "==> Fine-tuning detector on real data..."
python scripts/train_detector.py \
    --config configs/detector_finetune.yaml \
    --dataset_root real_data_work/dataset \
    --checkpoint   outputs/detector_best.pt \
    --save_path    outputs/detector_finetune.pt \
    --mix_synthetic

# ── Step 3: Train trajectory LSTM on synthetic data ──
# The synthetic LSTM training uses a camera distribution that now includes
# the corrected real camera range.  LSTM is not re-trained on real data because
# XYZ labels are unreliable until reconstruct_3d_from_2d is re-run with
# corrected intrinsics (done in Step 5).
echo "==> Training trajectory LSTM on synthetic data..."
python scripts/train_trajectory.py \
    --config configs/trajectory.yaml

# ── Step 4: Evaluate corrected pipeline on real data (3 test modes) ──
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
