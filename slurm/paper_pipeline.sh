#!/bin/bash
#SBATCH --job-name=paper-pipeline
#SBATCH --output=paper_pipeline_%j.out
#SBATCH --error=paper_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|l40s|a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ===========================================================================
# PAPER PIPELINE — full round for one model (run AFTER paper_gate.sh passes).
#
# Steps:
#   1. Disjoint prompt prep (resume-safe)         5. E1-B: clean add on benign (positive control, induce refusal)
#   2. Held-out direction extraction (resume)     6. E2:   attacked add on harmful (HEADLINE contrast)
#   3. Behavioral generation k=0/3/10 + benign    7. E1-A: clean ablate on harmful (re-confirm gate at scale)
#   4. Tracing + summaries + H3 association        8. print where results landed
#
# Default model = Llama-3.1-8B. For cross-model (E4), set MODEL_CONFIG and a
# distinct DIR_PATH so directions/results don't collide. See bottom of file.
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

MODEL_CONFIG="${MODEL_CONFIG:-}"            # e.g. "--model-config configs/model_qwen7b.yaml"
DIR_PATH="outputs/paper/directions/refusal_direction.pt"

python scripts/check_cuda_stack.py

# 1. Disjoint prompt sets (skip if already present).
python scripts/prepare_report7_disjoint_prompts.py --prefix paper --traced-n 150 --direction-n 50

# 2. Held-out refusal direction (resume-safe: skips if the .pt exists).
python scripts/extract_refusal_direction.py \
    --config configs/experiments/paper/extract_direction_paper.yaml $MODEL_CONFIG

# 3. Behavioral generation at scale (no Report-6 cache reuse).
python scripts/run_report7_generation.py \
    --config configs/experiments/paper/generation_paper.yaml $MODEL_CONFIG \
    --no-reuse-report6

# 4. Tracing + standardized summaries + prompt-level H3 association.
python scripts/run_report7_tracing.py \
    --config configs/experiments/paper/tracing_paper.yaml $MODEL_CONFIG \
    --direction-path "$DIR_PATH"
python scripts/summarize_report7_tracing.py \
    --config configs/experiments/paper/tracing_paper.yaml
python scripts/analyze_report7_prompt_association.py \
    --summary-dir outputs/paper/summaries \
    --condition harmful_k03 --layers 20 24 27

# 5. E1-B positive control: add direction on CLEAN benign -> should induce refusal.
python scripts/run_directional_intervention.py \
    --config configs/experiments/paper/e1_clean_add_benign.yaml $MODEL_CONFIG \
    --direction-path "$DIR_PATH"

# 6. E2 HEADLINE: same add intervention on ATTACKED harmful -> should NOT restore refusal.
python scripts/run_directional_intervention.py \
    --config configs/experiments/paper/e2_attacked_add_harmful.yaml $MODEL_CONFIG \
    --direction-path "$DIR_PATH"

# 7. E1-A re-confirm the gate at full scale (clean ablate on harmful).
python scripts/run_directional_intervention.py \
    --config configs/experiments/paper/e1_clean_ablate_harmful.yaml $MODEL_CONFIG \
    --direction-path "$DIR_PATH"

echo ""
echo "================ KEY RESULTS ================"
for f in outputs/paper/directional/*_summary.csv \
         outputs/paper/generations/report7_generation_summary.csv; do
    echo "--- $f ---"; cat "$f" 2>/dev/null || echo "(missing)"
done
echo "Tracing + H3: outputs/paper/summaries/"
echo "============================================="
date

# ---------------------------------------------------------------------------
# CROSS-MODEL (E4): re-run this script per model. Because the configs hardcode
# outputs/paper/directions/refusal_direction.pt, run one model at a time and
# move outputs/paper -> outputs/paper_<model> between runs, e.g.:
#   MODEL_CONFIG="--model-config configs/model_qwen7b.yaml" sbatch slurm/paper_pipeline.sh
#   # after it finishes:  mv outputs/paper outputs/paper_qwen7b
# (A future revision can parametrize the output root directly.)
# ---------------------------------------------------------------------------
