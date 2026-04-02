#!/bin/bash
#SBATCH --job-name=refusal-patching
#SBATCH --output=outputs/logs/patching_%j.out
#SBATCH --error=outputs/logs/patching_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------------------------------------------------------------------------
# Pilot patching experiment.
# Requires refusal direction to be precomputed.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
date

source ~/.bashrc
conda activate refusal-decay

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/logs outputs/patches

# Ensure direction is extracted first
DIRECTION_PATH="outputs/directions/refusal_direction.pt"
if [ ! -f "$DIRECTION_PATH" ]; then
    echo "Direction not found at $DIRECTION_PATH. Extracting first..."
    python scripts/extract_refusal_direction.py --config configs/base.yaml
fi

echo "Running patching experiment..."
python scripts/run_patching.py \
    --config configs/experiments/patching.yaml

echo "Generating patching comparison plot..."
python scripts/plot_results.py \
    --patch-dir outputs/patches \
    --plot-dir outputs/plots

echo "Done. Results in outputs/patches/"
date
