# Run Report 6

This runbook is the narrow, restartable path for COMPSCI 602 Report 6.

The Report 6 result blocks are:

1. Observational tracing under baseline vs prefilling
2. Small targeted patching under the attacked condition

The default Report 6 outputs land under `outputs/report6/`.

## Before You Start

- Make sure the conda environment is available:
  - `conda env create -f environment.yml`
  - `conda activate refusal-decay`
- Make sure `HF_TOKEN` is set for gated Llama access.
- Replace the placeholder prompt data in `data/harmful_prompts.jsonl` and `data/benign_prompts.jsonl` if you want reportable results.
- Use `--model-config configs/model_3b.yaml` for a smoke test.
- Use the default 8B config for the actual report run.

## Report 6 Configs

- Master generation config: `configs/experiments/report6/generation_report6.yaml`
- Individual generation reruns:
  - `configs/experiments/report6/baseline_k0.yaml`
  - `configs/experiments/report6/prefill_k3.yaml`
  - `configs/experiments/report6/prefill_k10.yaml`
- Direction extraction: `configs/experiments/report6/extract_direction.yaml`
- Tracing: `configs/experiments/report6/trace_report6.yaml`
- Patching: `configs/experiments/report6/patch_report6.yaml`

## Recommended Order

### 1. Generate baseline and attacked outputs

Full Report 6 generation block:

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml
```

This runs:

- harmful `k=0`
- harmful `k=3`
- harmful `k=10` if configured
- benign `k=0`

Outputs:

- `outputs/report6/generations/harmful_k00/classified.jsonl`
- `outputs/report6/generations/harmful_k03/classified.jsonl`
- `outputs/report6/generations/harmful_k10/classified.jsonl`
- `outputs/report6/generations/benign_k00/classified.jsonl`
- `outputs/report6/generations/report6_generation_combined.jsonl`
- `outputs/report6/generations/report6_generation_summary.csv`

Short-on-time version:

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml \
  --conditions harmful_k00 harmful_k03 benign_k00
```

### 2. Extract the refusal direction

```bash
python scripts/extract_refusal_direction.py \
  --config configs/experiments/report6/extract_direction.yaml
```

Output:

- `outputs/report6/directions/refusal_direction.pt`

### 3. Run observational tracing

```bash
python scripts/run_report6_tracing.py \
  --config configs/experiments/report6/trace_report6.yaml
```

Outputs:

- `outputs/report6/traces/traces_k00.parquet`
- `outputs/report6/traces/traces_k03.parquet`
- `outputs/report6/traces/traces_k10.parquet`
- `outputs/report6/traces/traces_all.parquet`

The default tracing run is the harmful prompt subset at layers `16, 20, 24, 27`.

### 4. Run targeted patching

```bash
python scripts/run_report6_patching.py \
  --config configs/experiments/report6/patch_report6.yaml
```

Default patching scope:

- attack condition: harmful `k=3`
- layers: `16, 24, 27`
- target positions: `1, 3, 5`
- source position: `-1`
- prompt subset: `10` harmful prompts

Outputs:

- `outputs/report6/patching/patch_layer16_ts-1_tt1.jsonl` and peers
- `outputs/report6/patching/patching_classified.jsonl`
- `outputs/report6/patching/patching_summary.csv`

### 5. Build report tables and machine-readable summaries

```bash
python scripts/summarize_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml
```

Outputs in `outputs/report6/summaries/`:

- `generation_refusal_rates_by_condition.csv`
- `generation_prompt_labels.csv`
- `trace_mean_projection_by_condition_layer.csv`
- `trace_projection_by_condition_layer_token.csv`
- `trace_prompt_level_key_layers.csv`
- `patching_refusal_recovery_summary.csv`
- `patching_prompt_results.csv`
- `report6_manifest.json`

### 6. Produce report-ready plots

```bash
python scripts/plot_report6_results.py \
  --config configs/experiments/report6/generation_report6.yaml
```

Outputs in `outputs/report6/plots/`:

- `report6_refusal_rate_by_condition.png`
- `report6_heatmap_baseline_vs_prefill.png`
- `report6_projection_vs_token_layer27.png`
- `report6_projection_vs_token_layer16.png`
- `report6_prompt_level_layer27_<prompt_id>.png`
- `report6_patching_comparison.png`

## Fastest Useful Path

If you are short on time, run this minimum path first:

1. `python scripts/run_report6_generation.py --config configs/experiments/report6/generation_report6.yaml --conditions harmful_k00 harmful_k03 benign_k00`
2. `python scripts/extract_refusal_direction.py --config configs/experiments/report6/extract_direction.yaml`
3. `python scripts/run_report6_tracing.py --config configs/experiments/report6/trace_report6.yaml`
4. `python scripts/summarize_report6_results.py --config configs/experiments/report6/generation_report6.yaml`
5. `python scripts/plot_report6_results.py --config configs/experiments/report6/generation_report6.yaml`

Then add patching:

6. `python scripts/run_report6_patching.py --config configs/experiments/report6/patch_report6.yaml`
7. Re-run summarize and plot

## Unity

### Interactive

```bash
srun --partition=gpu --gres=gpu:1 --mem=48G --time=08:00:00 --pty bash
source ~/.bashrc
conda activate refusal-decay
cd /path/to/refusal-decay
```

Then run the commands above in order.

### Batch

There is a single batch script:

```bash
sbatch slurm/report6_pipeline.sh
```

Edit the partition, GPU type, memory, time, and environment activation in that script before using it on Unity.

## Resume and Recovery

- All major scripts are restartable.
- Re-running the same command will reuse existing per-condition or per-`k` outputs when possible.
- Use `--no-resume` to force recomputation.
- If one step fails midway, fix the issue and rerun the same command.
- If you want a completely clean rerun of a block, remove only that block's subdirectory under `outputs/report6/` or rerun with `--no-resume`.

## What Each Step Is For

- Generation: behavioral baseline vs attack outcomes plus benign control behavior.
- Direction extraction: saves the refusal direction used by tracing and patching.
- Tracing: produces layer-by-token refusal-direction projections for harmful prompts.
- Patching: tests whether the late-layer attacked-state shift is causally relevant under a small targeted intervention grid.
- Summaries: builds the CSV tables used for the report.
- Plots: saves the minimum report-ready figures.

## Useful Variants

Smoke test on 3B:

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml \
  --model-config configs/model_3b.yaml \
  --conditions harmful_k00 harmful_k03 benign_k00
```

Single-condition rerun:

```bash
python scripts/run_report6_generation.py \
  --config configs/experiments/report6/generation_report6.yaml \
  --conditions harmful_k03
```

Force a clean patching rerun:

```bash
python scripts/run_report6_patching.py \
  --config configs/experiments/report6/patch_report6.yaml \
  --no-resume
```
