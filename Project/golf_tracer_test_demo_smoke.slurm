#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=tracer_smoke_test
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

python scripts/test_pipeline.py --config configs/pipeline_smoke.yaml --dataset_root demo_dataset --detector_ckpt outputs/detector_best.pt --trajectory_ckpt outputs/trajectory_best.pt --save_video --out_dir outputs/test_results --max_sequences 20


