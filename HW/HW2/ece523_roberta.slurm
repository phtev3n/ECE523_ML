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

# Install dependencies into the active Python environment
python -m pip install -q transformers accelerate papermill ipykernel

# Register this Python as a Jupyter kernel so papermill uses it
python -m ipykernel install --user --name pyenv --display-name "Python (pyenv)"

# Verify key imports before running the notebook
python -c "import torch; import transformers; import sklearn; print(f'torch={torch.__version__}, transformers={transformers.__version__}, GPU={torch.cuda.is_available()}')"

# Run the notebook using the pyenv kernel — --log-output sends cell prints to stdout
python -m papermill hw2_roberta.ipynb hw2_roberta.ipynb --no-progress-bar --log-output -k pyenv

# Clean up model checkpoints to save disk space
rm -rf ./roberta_syllogism

echo "Job finished: $(date)"
