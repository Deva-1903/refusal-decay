#!/bin/bash
#SBATCH --job-name=report6-pipeline
#SBATCH --output=report6_pipeline_%j.out
#SBATCH --error=report6_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
date

module load conda/latest
source ~/.bashrc
conda activate /work/pi_compsci602_umass_edu/devaanand_umass_edu/.conda/envs/refusal-decay

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    exit 1
fi

export HF_HOME="${HF_HOME:-/work/pi_compsci602_umass_edu/devaanand_umass_edu/.cache/huggingface}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/report6/logs

python scripts/check_cuda_stack.py

python scripts/run_report6_generation.py \
    --config configs/experiments/report6/generation_report6.yaml

python scripts/extract_refusal_direction.py \
    --config configs/experiments/report6/extract_direction.yaml

python scripts/run_report6_tracing.py \
    --config configs/experiments/report6/trace_report6.yaml

python scripts/run_report6_patching.py \
    --config configs/experiments/report6/patch_report6.yaml

python scripts/summarize_report6_results.py \
    --config configs/experiments/report6/generation_report6.yaml

python scripts/plot_report6_results.py \
    --config configs/experiments/report6/generation_report6.yaml

date
