# RUN_REPORT7 — Unity ops guide

End-to-end commands to (1) run the Report 7 pipeline on Unity, (2) commit the
right output files back to git, and (3) hand them back to Claude Code for
analysis.

The plan this doc executes is in `project_reports/report7_prompt_critique.md`
(TL;DR section). The SLURM script is `slurm/report7_pipeline.sh`.

---

## 0. Pre-flight (one time per Unity session)

```bash
# SSH into Unity and clone/pull the latest repo
ssh <unity-host>
cd /work/pi_compsci602_umass_edu/devaanand_umass_edu/refusal-decay
git pull --ff-only

# HF token (required for the gated Llama-3.1-8B model)
export HF_TOKEN=<your token>

# Optional: confirm the conda env is the same one the SLURM script expects
conda activate /work/pi_compsci602_umass_edu/devaanand_umass_edu/.conda/envs/refusal-decay
python scripts/check_cuda_stack.py
```

---

## 1. One-shot run (recommended)

Submit the full pipeline as one SLURM job. Runs steps 1–9 below in order.

```bash
sbatch slurm/report7_pipeline.sh
squeue -u $USER
tail -f report7_pipeline_<JOBID>.out
```

Typical wall time: **~6–7 hours** on a single A100. Breakdown:
- Held-out direction extraction: ~10 min
- Behavioral generation incl. k=10: ~15 min (or near-instant if R6 caches reused)
- Tracing: ~30 min
- Cross-condition patching (3 layers × 4 positions × 25 prompts): ~30 min
- Additive intervention on harmful (3 layers × 2 alphas × 3 positions × 25 prompts × [1 refusal + 5 random + 5 orthogonal] seeds): **~2.5 hr** (largest item)
- Additive benign positive control (same grid, k=0): **~2.5 hr**
- Stats / plots / verification: ~5 min

If wall time looks tight, see the **Fallback** section. The two
additive-intervention jobs are the cost; the cross-condition patching
result is the one experiment R7 cannot ship without.

---

## 2. Step-by-step (use these for partial reruns)

Every step below mirrors a block in `slurm/report7_pipeline.sh`. Run from repo
root with the conda env active.

### 2.1 Build disjoint prompt sets (CPU, instant)
Splits `data/{harmful,benign}_prompts.jsonl` into two non-overlapping subsets:
- `data/processed/report7_traced_*.jsonl` — first 25 (used by tracing/patching/intervention; same as R6's traced set)
- `data/processed/report7_direction_*.jsonl` — next 50 (held-out for direction extraction; **disjoint from traced**)

```bash
python scripts/prepare_report7_disjoint_prompts.py
```

### 2.2 Held-out refusal-direction extraction (~10 min on A100)
Re-extracts the direction on the disjoint set and writes
`outputs/report7/directions/refusal_direction.pt`.

```bash
python scripts/extract_refusal_direction.py \
    --config configs/experiments/report7/extract_direction_heldout.yaml \
    --no-resume
```

### 2.3 Behavioral generation (k=0, k=3, benign k=0, k=10 rerun)
Reuses valid R6 caches by default; recomputes any all-error caches.

```bash
python scripts/run_report7_generation.py \
    --config configs/experiments/report7/generation_report7.yaml
```

To force the k=10 rerun even if a cached value exists:
```bash
python scripts/run_report7_generation.py \
    --config configs/experiments/report7/generation_report7.yaml \
    --conditions harmful_k10 \
    --no-resume \
    --no-reuse-report6
```

### 2.4 Tracing (uses the held-out direction)
```bash
python scripts/run_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml
```

### 2.5 Trace summaries + prompt-level H3 association
```bash
python scripts/summarize_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml

python scripts/analyze_report7_prompt_association.py \
    --condition harmful_k03 --layers 20 24 27
```

### 2.6 Cross-condition patching (k=0 source → k=3 target)
```bash
python scripts/run_report7_cross_condition_patching.py \
    --config configs/experiments/report7/patching_report7.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt
```

### 2.7 Additive refusal-direction intervention (harmful, multi-seed controls)
```bash
python scripts/run_report7_additive_direction_intervention.py \
    --config configs/experiments/report7/additive_intervention_report7.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt
```
Outputs three files in `outputs/report7/interventions/`:
- `additive_direction_results.csv` — per-prompt per-seed rows
- `additive_direction_summary.csv` — per-cell-per-seed counts/rates
- `additive_direction_seed_aggregated.csv` — mean/std restoration_rate across seeds (this is the headline for random/orthogonal)

### 2.7b Benign positive-control (specificity test)
Adds α·refusal_direction to **benign** prompts at k=0. If the model now
refuses, the direction is over-broad; if it doesn't, it's harmful-specific.

```bash
python scripts/run_report7_additive_direction_intervention.py \
    --config configs/experiments/report7/additive_intervention_benign_control_report7.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt
```
Outputs go to `outputs/report7/interventions_benign_control/` with `_benign` suffix.

### 2.8 Bootstrap CIs + McNemar
```bash
python scripts/run_report7_stats.py
```
Writes per-cell CI/p-value tables to `outputs/report7/summaries/`. The script
auto-detects the benign positive-control file and produces a parallel
`additive_direction_stats_benign_control.csv`.

### 2.9 Comparison, plots, verification, classifier spot-check (template)
```bash
python scripts/compare_report6_report7_patching.py
python scripts/plot_report7_results.py
python scripts/verify_report7_outputs.py

# Generates outputs/report7/classifier_spot_check/spot_check.yaml with 20
# stratified items. You fill in `your_label:` for each, then run `score`.
python scripts/run_classifier_spot_check.py generate
```

### 2.10 Classifier spot-check (manual, after the SLURM job pulls back)
This step is the cheapest way to address Report 6's #1 self-flagged validity
threat (the phrase classifier). Runs locally — no GPU.

```bash
# 1. Pull the run results so spot_check.yaml is on your machine.
git pull

# 2. Open and edit each item. Replace `your_label: TODO` with one of:
#       refusal | compliance | ambiguous
$EDITOR outputs/report7/classifier_spot_check/spot_check.yaml

# 3. Score the agreement.
python scripts/run_classifier_spot_check.py score

# 4. Read the report. Commit the yaml + report + csv back so the writeup
#    can cite the agreement rate.
cat outputs/report7/classifier_spot_check/spot_check_report.md
git add outputs/report7/classifier_spot_check/
git commit -m "report7: classifier spot-check (n=20)"
```

Plan to spend ~30 minutes on the labeling. The reported flip rate goes
straight into the Threats-to-Validity section of the Report 7 writeup.

`verify_report7_outputs.py` prints COMPLETE / MISSING / SUSPICIOUS to stdout and
writes `VERIFY_REPORT7.md`. Exit code is non-zero if anything is missing.

---

## 3. What to commit back

After the job finishes on Unity, commit and push these from the Unity checkout
so I can pull and analyze locally:

```bash
git add \
    data/processed/report7_traced_harmful.jsonl \
    data/processed/report7_traced_benign.jsonl \
    data/processed/report7_direction_harmful.jsonl \
    data/processed/report7_direction_benign.jsonl \
    outputs/report7/ \
    VERIFY_REPORT7.md \
    report7_pipeline_*.out \
    report7_pipeline_*.err

git commit -m "report7: pipeline run on Unity (job <JOBID>)"
git push
```

The key files I will read for analysis:
- `outputs/report7/generations/report7_generation_summary.csv`
- `outputs/report7/summaries/trace_generated_token_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_prompt_level_key_layers.csv`
- `outputs/report7/summaries/prompt_projection_label_association.csv`
- `outputs/report7/summaries/prompt_projection_label_differences.csv`
- `outputs/report7/patching/cross_condition_patching_results.csv`
- `outputs/report7/patching/cross_condition_patching_summary.csv`
- `outputs/report7/interventions/additive_direction_results.csv`
- `outputs/report7/interventions/additive_direction_summary.csv`
- `outputs/report7/interventions/additive_direction_seed_aggregated.csv`
- `outputs/report7/interventions_benign_control/additive_direction_results_benign.csv`
- `outputs/report7/interventions_benign_control/additive_direction_summary_benign.csv`
- `outputs/report7/interventions_benign_control/additive_direction_seed_aggregated_benign.csv`
- `outputs/report7/summaries/cross_condition_patching_stats.csv`
- `outputs/report7/summaries/additive_direction_stats.csv`
- `outputs/report7/summaries/additive_direction_stats_seed_aggregated.csv`
- `outputs/report7/summaries/additive_direction_stats_benign_control.csv`
- `outputs/report7/summaries/additive_direction_stats_seed_aggregated_benign_control.csv`
- `outputs/report7/summaries/report7_intervention_stats_combined.csv`
- `outputs/report7/summaries/report6_vs_report7_intervention_comparison.csv`
- `outputs/report7/classifier_spot_check/spot_check.yaml` (template; you'll fill it in locally)
- `outputs/report7/plots/*.png`
- `VERIFY_REPORT7.md`
- `report7_pipeline_<JOBID>.out` / `.err` (for runtime + warning surface)

If anything in `outputs/report7/` is large (parquet traces ≥ 100MB), keep it on
Unity and commit only the CSV summaries. The combined parquet
`outputs/report7/traces/traces_all.parquet` is the largest file; commit it only
if smaller than ~50MB.

---

## 4. Fallback plan (compute pinch)

If the full job is going to overrun, in escalating order of pain:

1. **Skip benign positive-control (step 2.7b)**: comment out that block in `slurm/report7_pipeline.sh`. Saves ~2.5 hr. The specificity claim weakens but the headline causal result stays.
2. **Cut multi-seed from 5 to 3**: edit `additive_intervention_report7.yaml` and `additive_intervention_benign_control_report7.yaml`: `seeds: [42, 43, 44]`. Saves ~40% of additive runtime.
3. **Drop k=10**: `--conditions harmful_k00 harmful_k03 benign_k00` in step 2.3.
4. **Shrink the additive grid**: edit `additive_intervention_report7.yaml`:
   - `target_positions: [0, 1]`  (was [0, 1, 3])
   - `direction_types: [refusal, random]`  (drop orthogonal)
5. **Cross-condition patching only at 24/27**: edit `patching_report7.yaml`:
   - `layers: [24, 27]`  (drop 16)
6. **Cut prompts to 10**: pass `--max-prompts 10` to the patching and additive run scripts.

The single most-important experiment is **2.6 cross-condition patching**.
Everything else can be deferred to "future work" without losing the report.

---

## 5. Smoke test (3B model, 5 prompts, ~10 min on any GPU)

Use this to validate the pipeline before consuming an A100 hour.

```bash
python scripts/prepare_report7_disjoint_prompts.py --traced-n 5 --direction-n 10
python scripts/extract_refusal_direction.py \
    --config configs/experiments/report7/extract_direction_heldout.yaml \
    --model-config configs/model_3b.yaml --no-resume

python scripts/run_report7_generation.py \
    --config configs/experiments/report7/generation_report7.yaml \
    --model-config configs/model_3b.yaml \
    --conditions harmful_k00 harmful_k03 benign_k00

python scripts/run_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml \
    --model-config configs/model_3b.yaml

python scripts/summarize_report7_tracing.py \
    --config configs/experiments/report7/tracing_report7.yaml

python scripts/run_report7_cross_condition_patching.py \
    --config configs/experiments/report7/patching_report7.yaml \
    --model-config configs/model_3b.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt \
    --max-prompts 5

python scripts/run_report7_additive_direction_intervention.py \
    --config configs/experiments/report7/additive_intervention_report7.yaml \
    --model-config configs/model_3b.yaml \
    --direction-path outputs/report7/directions/refusal_direction.pt \
    --max-prompts 5

python scripts/run_report7_stats.py --n-boot 200
python scripts/verify_report7_outputs.py || true
```

Don't commit the smoke-test outputs — wipe `outputs/report7/` and re-run for
real with the 8B model afterwards.

---

## 6. After you push: what I'll do here

1. `git pull` and re-read the bullet list in section 3.
2. Diff `outputs/report7/summaries/` against `outputs/report6/summaries/` for the H2 monotone gradient — does it survive the held-out direction?
3. Read `prompt_projection_label_differences.csv` for H3 (refusers vs. compliers at layers 24/27).
4. Read `cross_condition_patching_summary.csv` + `cross_condition_patching_stats.csv` for H4 — does clean-source patching restore refusal? Do the bootstrap CIs exclude zero?
5. Read `additive_direction_summary.csv` + `additive_direction_stats.csv` + the seed-aggregated file for H5 (refusal-direction sufficiency) and H6 (refusal vs. multi-seed random vs. multi-seed orthogonal — is the refusal mean outside the random±SD band?).
6. Read the **benign positive-control** stats for the specificity claim — refusal direction shouldn't induce false-refusal on benign prompts. If it does (above the random/orthogonal noise floor), the specificity story changes.
7. Read `report6_vs_report7_intervention_comparison.csv` for the headline table.
8. Read `outputs/report7/classifier_spot_check/spot_check_report.md` once you've labeled the YAML — quote the auto-vs-human agreement rate in the threats-to-validity section.
9. Pick the headline framing (info-loss vs. active-suppression) per the interpretation rules in `project_reports/claude_code_prompt_report7_top_tier.md` §"Scientific interpretation rules", and draft Discussion bullets.
