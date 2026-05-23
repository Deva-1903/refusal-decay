#!/bin/bash
#SBATCH --job-name=paper-gate
#SBATCH --output=paper_gate_%j.out
#SBATCH --error=paper_gate_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|l40s|a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ===========================================================================
# PAPER GATE — the go/no-go decision before investing in the full round.
#
# Question this answers: does ablating the refusal direction at every
# layer/position actually BREAK refusal in the clean (k=0) setting?
#   - If YES (refusal rate drops sharply): the intervention works, the method
#     is sound, run paper_pipeline.sh next.
#   - If NO (refusal barely moves): the direction/hook is underpowered. Stop
#     and fix that before spending compute on the rest.
#
# Cheap: ~30-60 min on one GPU. Default model = Llama-3.1-8B (the configs'
# default). To gate a different model, set MODEL_CONFIG below.
# ===========================================================================

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
date

module load conda/latest
source ~/.bashrc
conda activate /work/pi_compsci602_umass_edu/devaanand_umass_edu/.conda/envs/refusal-decay

if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN is not set."; exit 1; fi
export HF_HOME="${HF_HOME:-/work/pi_compsci602_umass_edu/devaanand_umass_edu/.cache/huggingface}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/paper/logs

# Optional: gate a non-default model, e.g. MODEL_CONFIG="--model-config configs/model_qwen7b.yaml"
MODEL_CONFIG="${MODEL_CONFIG:-}"

python scripts/check_cuda_stack.py

# 1. Disjoint prompt sets at paper scale (150 traced + 50 held-out direction).
python scripts/prepare_report7_disjoint_prompts.py --prefix paper --traced-n 150 --direction-n 50

# 2. Held-out refusal direction.
python scripts/extract_refusal_direction.py \
    --config configs/experiments/paper/extract_direction_paper.yaml \
    $MODEL_CONFIG --no-resume

# 3. THE GATE: directional ablation on clean harmful prompts.
python scripts/run_directional_intervention.py \
    --config configs/experiments/paper/e1_clean_ablate_harmful.yaml \
    $MODEL_CONFIG \
    --direction-path outputs/paper/directions/refusal_direction.pt

echo ""
echo "================ GATE RESULT ================"
echo "Read outputs/paper/directional/e1_clean_ablate_harmful_summary.csv"
echo "PASS if intervened_refusal_rate << baseline_refusal_rate (ablation broke refusal)."
echo "FAIL if the two rates are similar (intervention underpowered — fix before full run)."
cat outputs/paper/directional/e1_clean_ablate_harmful_summary.csv 2>/dev/null || true
echo "============================================="
date
