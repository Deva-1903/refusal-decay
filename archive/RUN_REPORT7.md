# Run Report 7

This runbook prepares the Report 7 empirical upgrade: behavioral replication, tracing summaries, prompt-level association, cross-condition patching, additive direction intervention, plots, and verification.

## Unity Setup

Start or reattach your working session:

```bash
tmux ls
tmux attach -t refusal
```

If the session does not exist:

```bash
tmux new -s refusal
```

Go to the repo and load the environment:

```bash
cd /work/pi_compsci602_umass_edu/devaanand_umass_edu/refusal-decay
module load conda/latest
conda activate /work/pi_compsci602_umass_edu/devaanand_umass_edu/.conda/envs/refusal-decay
export HF_HOME=/work/pi_compsci602_umass_edu/devaanand_umass_edu/.cache/huggingface
export HF_TOKEN=hf_XXXXXXXXXXXXXXXX
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
```

Check Hugging Face access before GPU time:

```bash
huggingface-cli whoami
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'); print('Access OK')"
```

Request a modern GPU:

```bash
srun --partition=gpu --gres=gpu:1 --constraint=a100\|a40\|l40s --mem=48G --time=08:00:00 --pty bash
```

Inside the GPU shell, reload the env and check CUDA:

```bash
module load conda/latest
conda activate /work/pi_compsci602_umass_edu/devaanand_umass_edu/.conda/envs/refusal-decay
export HF_HOME=/work/pi_compsci602_umass_edu/devaanand_umass_edu/.cache/huggingface
export HF_TOKEN=hf_XXXXXXXXXXXXXXXX
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
cd /work/pi_compsci602_umass_edu/devaanand_umass_edu/refusal-decay
nvidia-smi -L
python scripts/check_cuda_stack.py
```

## Command Order

1. Verify or rerun behavioral outputs:

```bash
python scripts/run_report7_generation.py \
  --config configs/experiments/report7/generation_report7.yaml \
  --conditions harmful_k00 harmful_k03 benign_k00
```

2. Ensure the Report 7 refusal direction exists:

If the Report 6 direction was already produced for the same model and layer set, reuse it to save GPU time:

```bash
mkdir -p outputs/report7/directions
cp outputs/report6/directions/refusal_direction.pt outputs/report7/directions/refusal_direction.pt
```

Otherwise extract it fresh:

```bash
python scripts/extract_refusal_direction.py \
  --config configs/experiments/report7/generation_report7.yaml
```

3. If needed, run tracing with Report 7 paths:

```bash
python scripts/run_report6_tracing.py \
  --config configs/experiments/report7/tracing_report7.yaml \
  --direction-path outputs/report7/directions/refusal_direction.pt
```

If Report 6 traces are already good, you can skip rerunning tracing and still run the summary script. It falls back to `outputs/report6/traces`.

4. Build standardized tracing summaries:

```bash
python scripts/summarize_report7_tracing.py \
  --config configs/experiments/report7/tracing_report7.yaml
```

5. Run prompt-level association:

```bash
python scripts/analyze_report7_prompt_association.py
```

6. Run cross-condition patching:

```bash
python scripts/run_report7_cross_condition_patching.py \
  --config configs/experiments/report7/patching_report7.yaml \
  --direction-path outputs/report7/directions/refusal_direction.pt
```

7. Run additive direction intervention:

```bash
python scripts/run_report7_additive_direction_intervention.py \
  --config configs/experiments/report7/additive_intervention_report7.yaml \
  --direction-path outputs/report7/directions/refusal_direction.pt
```

8. Compare Report 6 and Report 7 interventions:

```bash
python scripts/compare_report6_report7_patching.py
```

9. Plot:

```bash
python scripts/plot_report7_results.py
```

10. Verify:

```bash
python scripts/verify_report7_outputs.py
```

## Expected Outputs

- `outputs/report7/generations/report7_generation_summary.csv`
- `outputs/report7/summaries/trace_generated_token_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_all_position_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_token_trajectory_by_condition_layer.csv`
- `outputs/report7/summaries/prompt_projection_label_association.csv`
- `outputs/report7/summaries/prompt_projection_label_differences.csv`
- `outputs/report7/patching/cross_condition_patching_results.csv`
- `outputs/report7/patching/cross_condition_patching_summary.csv`
- `outputs/report7/interventions/additive_direction_results.csv`
- `outputs/report7/interventions/additive_direction_summary.csv`
- `outputs/report7/summaries/report6_vs_report7_intervention_comparison.csv`
- `outputs/report7/plots/report7_*.png`
- `VERIFY_REPORT7.md`

## Fallback Plan

If runtime is tight:

- Skip `harmful_k10`.
- Keep prompt-level association, because it is cheap and high value.
- Run cross-condition patching first for layers `24` and `27`.
- Run target positions `0` and `1` first.
- Run additive intervention first for layer `27` and alpha `1.0` and `2.0`.
- Stay at `10` prompts unless the first run is fast.

## Highest-Value Minimal Run

If only one new empirical block can run, run cross-condition patching:

```bash
python scripts/run_report7_cross_condition_patching.py \
  --config configs/experiments/report7/patching_report7.yaml \
  --direction-path outputs/report7/directions/refusal_direction.pt
```

That directly addresses the weakest Report 6 causal test.
