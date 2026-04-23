#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_demo_video
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --time=02:00:00

# --------------------
### Code Section
# --------------------

python --version
pwd

# ── Step 1: Generate per-shot overlay videos (full learned pipeline) ──
echo "==> Running full learned pipeline with video output..."
python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/demo_videos

# ── Step 2: Compile all shot clips into a single demo reel ──
# Concatenates seq_*_overlay.mp4 files into demo_reel.mp4 via ffmpeg.
# Each 24-frame clip at 60 fps = 0.4s; 13 shots ≈ 5.2s total.
echo "==> Compiling demo reel..."
python scripts/compile_demo_reel.py \
    --in_dir outputs/demo_videos

echo "==> Done. Download outputs/demo_videos/demo_reel.mp4"
echo "    Individual shots also at: outputs/demo_videos/seq_*_overlay.mp4"
