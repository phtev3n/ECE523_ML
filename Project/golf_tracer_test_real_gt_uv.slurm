#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_test_real_gt_uv
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
# --------------------
### Code Section
# --------------------
python --version
pwd
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results_gt_uv --use_gt_uv

