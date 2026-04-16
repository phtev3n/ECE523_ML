#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=ece523_cnn_train
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
python scripts/make_demo_data.py --out_dir demo_dataset --num_sequences 500
python scripts/train_detector.py --config configs/detector.yaml --dataset_root demo_dataset
python scripts/train_trajectory.py --config configs/trajectory.yaml --dataset_root demo_dataset
python scripts/test_pipeline.py --config configs/pipeline_smoke.yaml --dataset_root demo_dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/test_results --max_sequences 20
