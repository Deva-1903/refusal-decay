# Report 6 Verification Summary

Audit date: 2026-05-04  
Auditor: internal pre-report audit  
Working directory: `refusal-decay/`

---

## 1. Current project scope for Report 6

Report 6 covers two experimental blocks:

1. **Observational tracing** — refusal-direction projection across layers and token positions under harmful baseline (k=0) and prefilling attack (k=3, k=10).
2. **Targeted patching** — causal intervention under the k=3 attack, testing whether injecting refusal-direction signal at early generated token positions restores refusal behavior.

Model: `meta-llama/Llama-3.1-8B-Instruct` (32 layers, hidden size 4096).  
Prompt data: real AdvBench (harmful, 520 records) and Alpaca (benign, 520 records).

---

## 2. What was found in the repo

### Infrastructure (all present)
- `configs/experiments/report6/` — 7 configs covering generation, direction extraction, tracing, patching, and per-condition overrides.
- `scripts/run_report6_generation.py`, `run_report6_tracing.py`, `run_report6_patching.py`, `summarize_report6_results.py`, `plot_report6_results.py` — all present and have correct logic.
- `slurm/report6_pipeline.sh` — batch job template exists.
- `src/probing/`, `src/patching/`, `src/generation/`, `src/classification/` — all present.

### Output tree under `outputs/report6/`
All seven subdirectories exist: `directions/`, `generations/`, `logs/`, `patching/`, `plots/`, `summaries/`, `traces/`.

### Report 3-era outputs (do not confuse with Report 6)
- `report3_runs/` and `report3_runs_1/` — older runs from 2026-04-06/07.
- These have different output schemas (`prompt_text`, `generated_text`, `prefix_text_used`) vs the Report 6 schema (`condition_name`, `dataset_name`).
- Report 3 behavioral data (generation + sweep) is valid and complete. It is **not** in the Report 6 pipeline and is treated as a separate reference below.

---

## 3. Behavioral experiments

### CRITICAL PROBLEM: All Report 6 generation outputs are error records

**Status: FAILED**

The `outputs/report6/generations/` classified JSONL files for all four conditions exist but contain only error records:

| Condition | n_total | n_valid | n_refusal | n_compliance | n_error |
|-----------|---------|---------|-----------|--------------|---------|
| harmful_k00 | 25 | 0 | 0 | 0 | 25 |
| harmful_k03 | 25 | 0 | 0 | 0 | 25 |
| harmful_k10 | 25 | 0 | 0 | 0 | 25 |
| benign_k00  | 25 | 0 | 0 | 0 | 25 |

**Root cause (confirmed from `outputs/report6/logs/report6_generation.log`):**

The generation script was first run on 2026-05-05 on a node with an incompatible GPU (`NVIDIA GeForce GTX 1080 Ti`, sm_61). Every prompt failed with:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

These failed records (with `refusal_phrase_label: "error"`) were saved to `classified.jsonl`. On 2026-05-06, the script was rerun on a node with an `NVIDIA L40S` (sm_89) which would have worked, but the resume logic saw the existing `classified.jsonl` files and loaded them from cache without regenerating. The generation summary and combined JSONL reflect only the error records.

**The generation has never successfully completed under the Report 6 pipeline.**

**What this affects:**
- `outputs/report6/generations/report6_generation_summary.csv` — all refusal_rate fields are empty/NaN.
- `outputs/report6/summaries/generation_refusal_rates_by_condition.csv` — same.
- `outputs/report6/summaries/generation_prompt_labels.csv` — no valid labels.
- `outputs/report6/plots/report6_refusal_rate_by_condition.png` — the file exists (35KB) but plots NaN bars (the bar chart is visually blank or shows 0% for all conditions). Do not use this plot in the report.

**Valid behavioral data DOES exist in report3_runs (different schema, different era):**

From `report3_runs/20260406_222202_8b_harmful_sweep_k0_3_10_25/report3_prefilling_sweep/sweep_classified.jsonl`:

| k | n | refusal | compliance | refusal_rate |
|---|---|---------|------------|--------------|
| 0 | 25 | 24 | 1 | 96% |
| 3 | 25 | 8 | 17 | 32% |
| 10 | 25 | 9 | 16 | 36% |

From `report3_runs/20260407_021031_8b_benign_k00_25/baseline_classified.jsonl`:

| k | n | refusal | compliance |
|---|---|---------|------------|
| 0 | 25 | 0 | 25 | (benign, all complied as expected) |

This data was generated on the same model and is reportable, but it lives outside the Report 6 pipeline and uses a different JSONL schema. The report must explicitly note these results come from the Report 3 era exploratory run, or the generation must be re-run under Report 6.

---

## 4. Tracing experiments

**Status: COMPLETE AND VALID**

### Direction extraction
- File: `outputs/report6/directions/refusal_direction.pt`
- Extracted from 50 harmful + 50 benign prompts.
- Layers covered: 16, 20, 24, 27.
- Direction vectors: shape `[4096]`, near-unit norm (0.997–1.001), dtype bfloat16.
- GPU used: NVIDIA L40S (sm_89). Log confirms clean completion.

### Tracing parquets

| File | Rows | Prompts | Layers | step range | projection mean |
|------|------|---------|--------|------------|-----------------|
| traces_k00.parquet | 1712 | 25 | [16,20,24,27] | 0–47 | +1.24 |
| traces_k03.parquet | 4596 | 25 | [16,20,24,27] | 0–47 | -2.09 |
| traces_k10.parquet | 4284 | 25 | [16,20,24,27] | 0–47 | -2.53 |
| traces_all.parquet  | 10592 | 25×3 | [16,20,24,27] | 0–47 | — |

Columns: `prompt_id, label, category, prefix_k, step, is_prefill, gen_token_pos, token_text, layer, projection`.

Note: there is no `token_position` column; the position columns are `step` (absolute) and `gen_token_pos` (relative to generation start, where -1 = prefill step).

### Signal validation

Mean projection by condition and layer (generated tokens only):

| Condition | Layer 16 | Layer 20 | Layer 24 | Layer 27 |
|-----------|----------|----------|----------|----------|
| k=0 | +1.46 | +1.95 | +1.19 | +0.38 |
| k=3 | +0.18 | -0.35 | -3.02 | -5.18 |
| k=10 | -0.24 | -0.79 | -3.42 | -5.69 |

This is exactly the expected pattern: under attack, the refusal-direction projection flips negative in late layers (24, 27) while early layers (16, 20) show weaker or no inversion at k=0, smaller decay at k=3. The k=10 projection is more negative than k=3 in all layers. The signal is strong and internally consistent.

The summary CSV `trace_mean_projection_by_condition_layer.csv` and `trace_projection_by_condition_layer_token.csv` (564 rows) are complete and correct.

---

## 5. Patching experiments

**Status: COMPLETE AND VALID**

### Configuration
- Attack condition: harmful_k03
- Layers: 16, 24, 27
- Target positions: 1, 3, 5
- Source position: -1
- Mode: replace
- Prompts: 10 harmful prompts

### Coverage: all 9 combinations ran cleanly

All 9 JSONL files exist (`patch_layer{L}_ts-1_tt{T}.jsonl` for L in {16,24,27}, T in {1,3,5}). Each has 10 records with text. No errors. GPU: L40S.

### Results summary (from `patching_classified.jsonl`, 90 records)

Under the k=3 attack (baseline_refusal_rate = 50% across the 10-prompt subset):

| Layer | Baseline refusal (out of 30) | Patched refusal (out of 30) | Restored | Lost |
|-------|------------------------------|------------------------------|----------|------|
| 16 | 15 | 15 | 0 | 0 |
| 24 | 15 | 13 | 0 | 2 |
| 27 | 15 | 14 | 0 | 1 |

**Interpretation:** Patching did not restore refusal in any combination. In two layer-24 combinations and one layer-27 combination, patching actually slightly reduced refusal (net delta: −1 to −2 cases). Layer 16 showed no effect. This is a consistent null-to-negative result for causal restoration.

The source projection at layer 27 is **negative** (mean: −0.46), which is consistent with the tracing finding that under k=3, the last-token (source_pos=-1) refusal-direction projection is already inverted at late layers. Injecting a negative projection at early target positions does not restore refusal — which is expected given the direction is already flipped.

The `patching_refusal_recovery_summary.csv` and `patching_prompt_results.csv` in `summaries/` correctly reflect these numbers.

### Important caveat
The individual raw patch files (`patch_layer*_ts-1_tt*.jsonl`) do NOT contain refusal labels — they only have baseline/patched text. Refusal labels and `refusal_restored`/`refusal_lost` flags are only present in `patching_classified.jsonl`. The summarize script correctly reads the classified file. Do not use the raw per-combination files for label-based analysis.

---

## 6. Figures and tables available for the report

### Required figures — status

| Figure | Status | File |
|--------|--------|------|
| 1. Refusal-rate table (baseline vs attack) | **DERIVABLE from report3 data, not from Report 6 pipeline** | Must be read from `report3_runs/20260406_222202.../sweep_classified.jsonl` |
| 2. Refusal-rate plot vs k (k=0,3,10) | **DERIVABLE from report3 data** | `report3/refusal_rate_vs_k.png` already exists (Report 3 era) |
| 3. Heatmap of projection by layer × token position | **AVAILABLE NOW** | `outputs/report6/plots/report6_heatmap_baseline_vs_prefill.png` (57KB, valid) |
| 4. Line plot for late layer (27) | **AVAILABLE NOW** | `outputs/report6/plots/report6_projection_vs_token_layer27.png` (107KB, valid) |
| 5. Line plot for comparison layer (16) | **AVAILABLE NOW** | `outputs/report6/plots/report6_projection_vs_token_layer16.png` (99KB, valid) |
| 6. Patching summary table | **AVAILABLE NOW** | `outputs/report6/summaries/patching_refusal_recovery_summary.csv` |
| 7. Patching comparison plot | **AVAILABLE NOW** | `outputs/report6/plots/report6_patching_comparison.png` (40KB, valid) |
| 8. Prompt-level raw plot (layer 27) | **AVAILABLE NOW** | `outputs/report6/plots/report6_prompt_level_layer27_advbench_0019.png` (70KB, valid) |

### Key problem
Figure 1 (refusal-rate table) and Figure 2 (refusal-rate plot vs k) **cannot be generated from the Report 6 pipeline as currently cached** because all generation outputs are error records. The Report 6 refusal-rate plot file (`report6_refusal_rate_by_condition.png`) exists but shows blank/NaN bars and must not be used.

The Report 3 era data has the correct behavioral numbers and is reportable, but the student must decide whether to:

- (A) Use the Report 3 numbers directly and cite them as from the Report 3 pilot run, or
- (B) Re-run generation under Report 6 (`--no-resume` flag on L40S) to get numbers inside the Report 6 pipeline.

Option B takes roughly 30–60 minutes on L40S for 3 conditions × 25 prompts at 64 tokens.

---

## 7. Can I start writing Report 6?

**Partially yes.** The mechanistic story (tracing + patching) is fully complete and valid. All tracing plots, heatmaps, and patching results can be written up now.

**One blocker:** The behavioral section (refusal rate by condition, k-sweep table) cannot be written using Report 6 pipeline outputs. Either re-run generation or explicitly cite the Report 3 numbers.

Do not present the existing `report6_refusal_rate_by_condition.png` in the report — it is a blank-bar chart produced from NaN data.

---

## 8. Exact next commands to run

### Option A: Re-run generation to fix the behavioral block (recommended)

On a Unity node with `--constraint=a100|a40|l40s`:

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml \
  --conditions harmful_k00 harmful_k03 benign_k00 \
  --no-resume
```

Then re-run summarize and plot:

```bash
python scripts/summarize_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml

python scripts/plot_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml
```

This will overwrite the error-record classified files and produce a valid `generation_refusal_rates_by_condition.csv` and a real refusal-rate bar chart.

### Option B: Use Report 3 behavioral data without re-running

Read refusal rates directly from:

```
report3_runs/20260406_222202_8b_harmful_sweep_k0_3_10_25/report3_prefilling_sweep/sweep_classified.jsonl
report3_runs/20260407_021031_8b_benign_k00_25/baseline_classified.jsonl
```

Key numbers: k=0: 24/25 refused (96%), k=3: 8/25 refused (32%), k=10: 9/25 refused (36%), benign k=0: 0/25 refused (0%).

These results come from the same model (Llama-3.1-8B-Instruct) and the same real AdvBench/Alpaca data, just a different run session (April 2026). Cite as "Report 3 pilot run" if using this option.

### Nothing else needs to be run

- Direction extraction: complete.
- Tracing: complete for k=0, k=3, k=10 over 25 prompts at layers 16, 20, 24, 27.
- Patching: complete for all 9 layer × target-position combinations.
- Summaries and plots (except refusal-rate): all valid and saved.

---

## 9. What should be deferred to Report 7

Per the original scope specification, the following are explicitly out of scope for Report 6:

- Larger patching grids (more layers, more source positions, more prompts).
- Suffix attacks.
- Attention-head or neuron-level analyses.
- Broad factorial sweeps over attack parameters.
- Cross-prompt patching (currently all patching is same-prompt self-patching).
- Truly held-out direction extraction (direction extraction currently uses the same 25 prompts as tracing).
- Benign-prompt tracing (only harmful prompts were traced).
- Guard-model evaluation (LlamaGuard integration not implemented; all refusal labels are phrase-list heuristics).

---

## 10. Data integrity notes

### Prompt data
- `data/harmful_prompts.jsonl`: 520 records, source=advbench, real AdvBench content confirmed.
- `data/benign_prompts.jsonl`: 520 records, source=alpaca, real Alpaca content confirmed.
- No synthetic placeholder records remain.

### Schema mismatch between Report 3 and Report 6 outputs
Report 3 outputs use keys: `prompt_text`, `generated_text`, `prefix_text_used`, `prefix_k` (string).  
Report 6 outputs use keys: `condition_name`, `dataset_name`, `error` (when failed), `prefix_k` (string).  
Do not mix these schemas in any aggregation script.

### Patching signal note
The source projection at layer 27 under k=3 is −0.46 (mean across 10 prompts at source_pos=-1). This means the refusal direction is **already pointing the wrong way** at the source position. Patching a negative projection into early positions cannot be expected to restore refusal — this explains the null result and is an important mechanistic finding to report.

---

## 11. Final recommendation

**RUN ONE THING FIRST, THEN WRITE**

The only missing piece for a complete Report 6 is valid behavioral generation data inside the Report 6 pipeline. The generation failed on a bad GPU on 2026-05-05 and the resume logic on 2026-05-06 (L40S) loaded the cached error files without regenerating.

**Run this on L40S:**

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml \
  --conditions harmful_k00 harmful_k03 benign_k00 \
  --no-resume

python scripts/summarize_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml

python scripts/plot_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml
```

After that, all must-have Report 6 outputs will be present and valid. Everything else — direction, traces, patching, summaries, heatmaps, line plots, prompt-level plot — is already verified complete.

If re-running generation is not feasible before the report deadline, use the Report 3 behavioral numbers (cited explicitly) and write the mechanistic sections now.
