#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_test_real
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

# Pass 1 — GT UV ceiling: bypasses detector, tests LSTM on real 3D with known-good 2D positions.
# If rmse_3d is still high here, the issue is camera calibration / 3D reconstruction, not the detector.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results_gt_uv --use_gt_uv

# Pass 2 — Classical background-subtraction detector: no domain gap, no learned model.
# Sets a 2D tracking baseline without synthetic-to-real transfer.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results_classical --classical_detector

# Pass 3 — Full learned pipeline: end-to-end with finetuned detector.
# Delta vs Pass 2 reveals the domain gap cost of the learned detector.
python scripts/test_pipeline.py --config configs/pipeline.yaml --dataset_root real_data_work/dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/real_test_results

