#!/bin/bash
#SBATCH --job-name=report7-pipeline
#SBATCH --output=report7_pipeline_%j.out
#SBATCH --error=report7_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Report 7 — full pipeline.
#
# Order is the order from project_reports/report7_prompt_critique.md:
#   1. Disjoint prompt prep         (CPU-only, instant)
#   2. Held-out direction extract   (~10 min)
#   3. Behavioral generation incl. k=10 rerun
#   4. Tracing on the held-out direction
#   5. Trace summaries + prompt-level H3 association
#   6. Cross-condition patching (k=0 -> k=3)
#   7. Additive direction intervention (with random + orthogonal controls)
#   8. Bootstrap CIs + McNemar
#   9. R6 vs R7 comparison + plots + verification

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
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

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/report7/logs

python scripts/check_cuda_stack.py

# ---------------------------------------------------------------------------
# 1. Disjoint prompt prep (CPU-only, must run before extraction)
# ---------------------------------------------------------------------------
python scripts/prepare_report7_disjoint_prompts.py

# ---------------------------------------------------------------------------
# 2. Held-out refusal-direction extraction (disjoint from traced prompts)
# ---------------------------------------------------------------------------
python scripts/extract_refusal_direction.py \
    --config configs/experiments/report7/extract_direction_heldout.yaml \
    --no-resume

# ---------------------------------------------------------------------------
# 3. Behavioral generation (k=0, k=3, benign k=0, k=10 rerun)
# ---------------------------------------------------------------------------
python scripts/run_report7_generation.py \
    --config configs/experiments/report7/generation_report7.yaml

# ---------------------------------------------------------------------------
# 4. Tracing on the held-out direction
# ---------------------------------------------------------------------------
python scripts/run_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml

# ---------------------------------------------------------------------------
# 5. Standardized tracing summaries (now includes prompt-level for H3)
# ---------------------------------------------------------------------------
python scripts/summarize_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml

python scripts/analyze_report7_prompt_association.py \
    --condition harmful_k03 \
    --layers 20 24 27

# ---------------------------------------------------------------------------
# 6. Cross-condition patching (k=0 source -> k=3 target)
# ---------------------------------------------------------------------------
python scripts/run_report7_cross_condition_patching.py \
    --config configs/experiments/report7/patching_report7.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt

# ---------------------------------------------------------------------------
# 7. Additive refusal-direction intervention (refusal + random + orthogonal)
# ---------------------------------------------------------------------------
python scripts/run_report7_additive_direction_intervention.py \
    --config configs/experiments/report7/additive_intervention_report7.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt

# ---------------------------------------------------------------------------
# 8. Bootstrap CIs + McNemar on per-prompt cell outcomes
# ---------------------------------------------------------------------------
python scripts/run_report7_stats.py

# ---------------------------------------------------------------------------
# 9. Cross-report comparison, plots, verification
# ---------------------------------------------------------------------------
python scripts/compare_report6_report7_patching.py
python scripts/plot_report7_results.py
python scripts/verify_report7_outputs.py || true

date
echo "Report 7 pipeline complete. See VERIFY_REPORT7.md for status."
