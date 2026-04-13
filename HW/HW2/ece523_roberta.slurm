#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=ece523_roberta
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=gpu_standard
#SBATCH --time=01:00:00

# --------------------
### Code Section
# --------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pwd

# Install any missing dependencies
python -m pip install -q datasets jupyter nbconvert

# Run the notebook in-place (executes all cells, saves outputs into the .ipynb)
python -m jupyter nbconvert --to notebook --execute hw2_roberta.ipynb --output hw2_roberta.ipynb --ExecutePreprocessor.timeout=1800

echo "Job finished: $(date)"
