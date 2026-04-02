#!/bin/bash
#SBATCH --job-name=refusal-baseline
#SBATCH --output=outputs/logs/baseline_%j.out
#SBATCH --error=outputs/logs/baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------------------------------------------------------------------------
# Baseline generation: normal inference on harmful prompts (k=0).
# Adjust partition, mem, and time to match your cluster config.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Environment setup ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
date

# Load conda environment (adjust name and path to match your cluster)
source ~/.bashrc
conda activate refusal-decay

# Set HF token from environment (set in your .bashrc or pass via sbatch --export)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before submitting this job."
    exit 1
fi

# Optional: use local model cache to avoid re-downloading
# export HF_HOME=/path/to/shared/hf_cache

# Move to repo root
cd "$SLURM_SUBMIT_DIR"

# Create output directories
mkdir -p outputs/logs outputs/generations/baseline

echo "Starting baseline generation..."
python scripts/run_baseline.py \
    --config configs/experiments/baseline.yaml

echo "Done. Results in outputs/generations/baseline/"
date
