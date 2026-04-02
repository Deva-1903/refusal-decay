#!/bin/bash
#SBATCH --job-name=refusal-tracing
#SBATCH --output=outputs/logs/tracing_%j.out
#SBATCH --error=outputs/logs/tracing_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------------------------------------------------------------------------
# Tracing: project residual stream onto refusal direction per token.
# This runs AFTER extract_refusal_direction.py has completed.
# More memory than baseline because we collect activations per step.
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
mkdir -p outputs/logs outputs/traces outputs/directions

# Step 1: Extract refusal direction (idempotent — skips if already exists)
echo "Step 1: Extracting refusal direction..."
python scripts/extract_refusal_direction.py \
    --config configs/base.yaml

# Step 2: Run tracing
echo "Step 2: Running projection tracing..."
python scripts/run_tracing.py \
    --config configs/experiments/tracing.yaml

# Step 3: Generate plots
echo "Step 3: Generating plots..."
python scripts/plot_results.py \
    --trace-dir outputs/traces \
    --plot-dir outputs/plots

echo "Done. Traces in outputs/traces/, plots in outputs/plots/"
date
