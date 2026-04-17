#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_train_real.slurm
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

# --------------------
### Code Section
# --------------------

python --version
pwd

# Fine-tune detector on real data, starting from the synthetically pre-trained checkpoint
python scripts/train_detector.py --config configs/detector.yaml --dataset_root real_data_work/dataset --checkpoint outputs/detector_best.pt

# Pass 1 — GT UV ceiling: bypasses detector, tests LSTM on real 3D with known-good 2D positions.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results_gt_uv --use_gt_uv

# Pass 2 — Classical background-subtraction detector: no domain gap, no learned model.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results_classical --classical_detector

# Pass 3 — Full learned pipeline with fine-tuned detector.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results
