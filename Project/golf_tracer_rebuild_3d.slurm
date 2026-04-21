#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=golf_tracer_rebuild_3d
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=ece523-2026sp
#SBATCH --partition=standard
#SBATCH --time=01:00:00

# --------------------
### Code Section
# --------------------
# PURPOSE: Re-run 3D trajectory reconstruction using corrected camera intrinsics
# (cx=256, cy=256 instead of the bad checkerboard cy=15.7) and rebuild the
# dataset so the LSTM has valid XYZ labels for real data fine-tuning.
#
# Run AFTER golf_tracer_train_real.slurm succeeds with the fixed detector,
# then re-run golf_tracer_train_real.slurm once more to fine-tune the LSTM.

python --version
pwd

# ── Step 1: Re-run 3D reconstruction with corrected intrinsics ──
# Collects all annotations_2d paths from shot_map.json using Python JSON parsing.
echo "==> Reconstructing 3D trajectories with corrected intrinsics..."
python - <<'PYEOF'
import json, subprocess, sys
from pathlib import Path

with open("real_data_work/shot_map.json") as f:
    shot_map = json.load(f)

ann_paths = [entry["annotations_2d"] for entry in shot_map]
missing = [p for p in ann_paths if not Path(p).exists()]
if missing:
    print(f"WARNING: {len(missing)} annotation files not found:")
    for p in missing:
        print(f"  {p}")
    ann_paths = [p for p in ann_paths if Path(p).exists()]

print(f"Reconstructing {len(ann_paths)} shots...")
cmd = [
    sys.executable, "scripts/reconstruct_3d_from_2d.py",
    "--camera_params",     "real_data_work/camera_params.json",
    "--camera_distance_m", "3.0",   # typical iPhone-to-ball distance for golf videos
    "--fps",               "60.0",
    "--out",               "real_data_work/trajectories_corrected.json",
] + ["--annotations_2d"] + ann_paths

ret = subprocess.run(cmd)
sys.exit(ret.returncode)
PYEOF

# ── Step 2: Rebuild dataset with corrected 3D trajectories and corrected intrinsics ──
echo "==> Rebuilding dataset with corrected 3D labels..."
python scripts/build_dataset.py \
    --shot_map       real_data_work/shot_map.json \
    --trajectories   real_data_work/trajectories_corrected.json \
    --camera_params  real_data_work/camera_params.json \
    --out_dir        real_data_work/dataset \
    --fps            60.0 \
    --reproj_threshold 20.0

# ── Step 3: Train trajectory LSTM on corrected real data ──
# Now that XYZ labels are physically consistent with the 2D annotations
# (using correct intrinsics), the LSTM can learn from real sequences.
echo "==> Training trajectory LSTM with corrected real XYZ labels..."
python scripts/train_trajectory.py \
    --config configs/trajectory.yaml \
    --dataset_root real_data_work/dataset \
    --checkpoint outputs/trajectory_best.pt

# ── Step 4: Final end-to-end test ──
echo "==> Final pipeline test with corrected 3D reconstruction..."
python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/real_test_results_final \
    --use_gt_uv

python scripts/test_pipeline.py \
    --config          configs/pipeline.yaml \
    --dataset_root    real_data_work/dataset \
    --detector_ckpt   outputs/detector_finetune_best.pt \
    --trajectory_ckpt outputs/trajectory_best.pt \
    --save_video \
    --out_dir         outputs/real_test_results_final_learned

echo "==> Rebuild complete. Check real_data_work/dataset/build_report.json"
echo "    Target: reprojection error mean < 20px."
echo "    If still high, increase --camera_distance_m (try 2.0 to 5.0 range)."
