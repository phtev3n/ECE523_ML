#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=ece523_cnn_test
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
# --------------------
### Code Section
# --------------------
python --version
pwd
python scripts/test_pipeline.py   --config configs/pipeline.yaml   --dataset_root demo_dataset   --detector_ckpt outputs/detector_best.pt   --trajectory_ckpt outputs/trajectory_best.pt   --save_video
