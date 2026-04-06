# Refusal Decay

**Tracing the positional decay of the refusal direction in safety-aligned LLMs under prefilling attacks.**

COMPSCI 602 — Mechanistic Interpretability / Causal Analysis of LLM Safety Behavior.

---

## Research Questions

This project investigates how the *refusal direction* — a direction in the residual stream that distinguishes harmful from benign representations — evolves token-by-token during generation when the model is subjected to a **prefilling attack** (forced compliant prefix tokens).

Specific questions:
1. Does the refusal-direction projection decline with token position during generation?
2. Does a larger forced prefix length k accelerate or deepen that decline?
3. Is the positional decay concentrated in particular layers?
4. Does patching the early-token refusal-direction component into later positions causally increase refusal probability?

These questions are grounded in the finding (Arditi et al., 2024) that refusal is mediated by a single direction in the residual stream, and extend it to the *temporal* (token-position) dimension under attack conditions.

---

## Model

| Role | Model |
|------|-------|
| **Research target** (course project results) | `meta-llama/Llama-3.1-8B-Instruct` |
| Smoke-test / debug / fast iteration | `meta-llama/Llama-3.2-3B-Instruct` |

The 8B model is the default in all configs. Use `--model-config configs/model_3b.yaml` to override for smoke tests. Do not report 3B results as the primary experimental findings.

---

## Scope

| Aspect | In scope | Out of scope (now) |
|--------|----------|---------------------|
| Model | Llama-3.1-8B (primary), Llama-3.2-3B (debug) | Other model families |
| Attack | Prefilling (k forced tokens) | Adversarial suffixes, GCG |
| Mechanism | Refusal direction (residual stream) | Attention heads, neurons |
| Probe | Difference-in-means | DAS, linear probes, nonlinear |
| Evaluation | Phrase-list classifier (heuristic) | Llama Guard (not yet integrated) |
| Experiments | Baseline, sweep, tracing, patching | Full causal ablation, cross-layer, multi-direction |

---

## Repository Structure

```
refusal-decay/
├── configs/
│   ├── base.yaml               # all defaults — 8B is the research target
│   ├── model_3b.yaml           # smoke-test override (use with --model-config)
│   ├── model_8b.yaml           # explicit reference for 8B
│   └── experiments/
│       ├── baseline.yaml
│       ├── prefilling_sweep.yaml
│       ├── tracing.yaml
│       └── patching.yaml
│
├── data/
│   ├── README.md               # how to get and normalize real benchmark data
│   ├── harmful_prompts.jsonl   # PLACEHOLDER — replace with AdvBench
│   └── benign_prompts.jsonl    # PLACEHOLDER — replace with Alpaca subset
│
├── src/
│   ├── config.py               # YAML loading + dot-access Config
│   ├── data/
│   │   ├── schema.py           # Prompt dataclass
│   │   └── loader.py           # JSONL loading and normalization
│   ├── generation/
│   │   ├── generator.py        # model loading + generation pipeline
│   │   └── prefilling.py       # prefilling attack (appends k prefix tokens)
│   ├── classification/
│   │   └── refusal_classifier.py  # phrase-list baseline + GuardClassifier interface
│   ├── probing/
│   │   ├── direction.py        # difference-in-means refusal direction
│   │   └── tracing.py          # per-token projection tracing during generation
│   ├── patching/
│   │   └── patch.py            # pilot direction-component patching
│   └── utils/
│
├── scripts/
│   ├── normalize_prompts.py    # convert external datasets to repo schema
│   ├── run_baseline.py
│   ├── run_prefilling_sweep.py
│   ├── extract_refusal_direction.py
│   ├── run_tracing.py
│   ├── run_patching.py
│   └── plot_results.py
│
├── slurm/
│   ├── baseline.sh
│   ├── tracing.sh
│   └── patching.sh
│
└── tests/                      # 40+ tests, no GPU required
```

---

## Installation

### Local

```bash
git clone <repo>
cd refusal-decay

conda env create -f environment.yml
conda activate refusal-decay

# Set HuggingFace token (required for gated Llama models)
cp .env.example .env
# Edit .env: HF_TOKEN=hf_...
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

### Cluster (SLURM)

```bash
git clone <repo> && cd refusal-decay
conda env create -f environment.yml
echo 'export HF_TOKEN=hf_...' >> ~/.bashrc && source ~/.bashrc
```

---

## Before Running Experiments: Prepare Real Data

The placeholder data in `data/` is for pipeline smoke-testing only. Replace it with real benchmark data before running any experiments you intend to report.

See [data/README.md](data/README.md) for full instructions. Quick version:

```bash
# 1. Download AdvBench harmful behaviors
wget https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv

# 2. Normalize into repo schema
python scripts/normalize_prompts.py \
    --input harmful_behaviors.csv \
    --text-col goal --label harmful \
    --source advbench --output data/harmful_prompts.jsonl

# 3. Get benign prompts from Alpaca (requires: pip install datasets)
python -c "
from datasets import load_dataset; import json
ds = load_dataset('tatsu-lab/alpaca', split='train')
with open('data/benign_prompts.jsonl','w') as f:
    for i, ex in enumerate(ds.select(range(500))):
        f.write(json.dumps({'prompt_id':f'alpaca_{i:04d}','text':ex['instruction'],
                            'label':'benign','category':'instruction','source':'alpaca'})+'\n')
"
```

---

## Running Experiments

### Test the pipeline first (no GPU, no real model)

```bash
pytest tests/ -v
```

All 40+ tests pass on CPU only. This validates config loading, data normalization, direction math, and classifier schema.

### Smoke-test with 3B (fast, ~15 min on a single GPU)

Always validate the full pipeline on the 3B model before committing GPU hours to 8B.

```bash
# Baseline
python scripts/run_baseline.py --model-config configs/model_3b.yaml

# Prefilling sweep
python scripts/run_prefilling_sweep.py --model-config configs/model_3b.yaml

# Extract direction + run tracing
python scripts/extract_refusal_direction.py --model-config configs/model_3b.yaml
python scripts/run_tracing.py --model-config configs/model_3b.yaml

# Plots
python scripts/plot_results.py
```

### Production run with 8B (research target)

Once the pipeline is validated and real data is in place:

```bash
# 8B is the default — no --model-config needed
python scripts/run_baseline.py
python scripts/run_prefilling_sweep.py
python scripts/extract_refusal_direction.py
python scripts/run_tracing.py
python scripts/run_patching.py
python scripts/plot_results.py
```

All scripts are **resume-safe**: re-running skips already-completed output files.
Use `--no-resume` to force a full rerun.

---

## Swapping Models

```bash
# Any script: override to 3B for smoke test
python scripts/run_baseline.py --model-config configs/model_3b.yaml

# Explicit 8B (same as default)
python scripts/run_baseline.py --model-config configs/model_8b.yaml
```

---

## Running on a SLURM Cluster

```bash
sbatch slurm/baseline.sh
# Wait for completion, then:
sbatch slurm/tracing.sh   # includes direction extraction
sbatch slurm/patching.sh
```

**Before submitting**, edit the SLURM scripts to match your cluster:
- `--partition` — your GPU partition name
- `--gres=gpu:1` — adjust to e.g. `gpu:a100:1`
- Memory: 8B requires ~24GB VRAM (increase `--mem` if needed)
- `conda activate refusal-decay` — or use modules

For interactive debugging:
```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
conda activate refusal-decay
python scripts/run_baseline.py --model-config configs/model_3b.yaml
```

---

## Evaluation Notes

### Refusal classifier

The pipeline uses a **phrase-list heuristic** classifier as its primary evaluation metric. It checks whether phrases like "I cannot", "I won't", "I'm sorry, but" appear in the first 200 characters of the response.

This is fast and requires no extra model, but it is a heuristic with known limitations:
- **False positives**: "I cannot stress enough how important..." classified as refusal
- **False negatives**: unusual refusal phrasing not in the list

**Llama Guard integration is not yet implemented.** The `GuardClassifier` interface in `src/classification/refusal_classifier.py` provides an adapter stub. Output records reserve `safety_guard_label`, `guard_model_name`, and `disagreement_flag` fields (currently `null`) for when guard evaluation is added.

Validate a sample of classifications manually before reporting final results.

### Patching experiment interpretation

The patching hypothesis is directional:

> If we restore the early-token refusal-direction component at later generation positions, the model should be *more likely to refuse*.

The expected result is **compliance → refusal** (patching restores refusal).
If patching has no effect, the refusal decision was already committed earlier.
If patching causes **refusal → compliance** (the wrong direction), that would be surprising and should be investigated separately.

The output metric in `run_patching.py` reports three cases:
- `refusal_restored` (compliance → refusal): expected positive causal effect
- `refusal_lost` (refusal → compliance): unexpected direction
- no change

---

## Output Format

### Generations (`outputs/generations/`)

JSONL, one record per prompt per k value:
```json
{
  "prompt_id": "advbench_0001",
  "label": "harmful",
  "category": "harmful_behavior",
  "source": "advbench",
  "prompt_text": "...",
  "prefix_k": 3,
  "generated_text": "...",
  "n_generated_tokens": 142,
  "refusal_phrase_label": "refusal",
  "matched_phrase": "I cannot",
  "refusal_classifier_version": "phrase_list_v1",
  "safety_guard_label": null,
  "guard_model_name": null,
  "disagreement_flag": null
}
```

### Traces (`outputs/traces/`)

Tidy Parquet/CSV, one row per (prompt, k, layer, token_position):
```
prompt_id | label | prefix_k | step | layer | projection
```

### Directions (`outputs/directions/`)

PyTorch `.pt`: `{layer_idx: unit_direction_tensor(hidden_size)}`

### Patches (`outputs/patches/`)

JSONL with `baseline_text`, `patched_text`, `baseline_phrase_label`, `patched_phrase_label`, `refusal_restored`, `refusal_lost` per record.

---

## First 24 Hours Plan

Work in this exact order — each step validates the previous one.

**Hours 1–2: Environment + tests**
- [ ] `conda env create -f environment.yml && conda activate refusal-decay`
- [ ] `pytest tests/ -v` — all 40+ tests should pass (no GPU needed)
- [ ] Confirm HF token works: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"`

**Hours 2–3: Get real data**
- [ ] Download AdvBench (`harmful_behaviors.csv`) and normalize (see above)
- [ ] Get Alpaca benign prompts and normalize
- [ ] Verify: `python -c "from src.data.loader import load_harmful_prompts; p=load_harmful_prompts('data/harmful_prompts.jsonl'); print(len(p), p[0])"`

**Hours 3–6: Smoke-test on 3B**
- [ ] `python scripts/run_baseline.py --model-config configs/model_3b.yaml`
- [ ] Open `outputs/generations/baseline/k00.jsonl` — read 3–5 outputs manually. Does the 3B model refuse harmful prompts?
- [ ] `python scripts/run_prefilling_sweep.py --model-config configs/model_3b.yaml`
- [ ] Does refusal rate drop as k increases? (Expected: yes, significantly by k=5)

**Hours 6–9: Direction extraction + tracing on 3B**
- [ ] `python scripts/extract_refusal_direction.py --model-config configs/model_3b.yaml`
- [ ] Check: direction file loads, norm ≈ 1.0 for each layer
- [ ] `python scripts/run_tracing.py --model-config configs/model_3b.yaml`
- [ ] `python scripts/plot_results.py`
- [ ] Does `projection_vs_position_layer*.png` show declining projection with token position for k=0? Does higher k start lower?

**Hours 9–12: Patching pilot on 3B**
- [ ] `python scripts/run_patching.py --model-config configs/model_3b.yaml`
- [ ] Check: does patching increase refusal rate at any (layer, target_position) combination?

**Hours 12–16: Rerun on 8B (if compute is available)**
- [ ] Repeat the sweep and tracing experiments with 8B (default config, no model-config flag)
- [ ] Compare 3B vs 8B results — do the patterns hold?

**Hours 16–24: Analysis + writing**
- [ ] Generate all final plots with 8B results
- [ ] Annotate patterns in `RESEARCH_LOG.md`
- [ ] Draft progress report: methods, preliminary results, figures, open questions

---

## Common Failure Modes

### 1. HF token not set or lacks model access
```
OSError: You are trying to access a gated repo...
```
Fix: `export HF_TOKEN=hf_...` — get token from huggingface.co/settings/tokens.
Also: request access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace if not already approved.

### 2. CUDA OOM (especially 8B)
```
torch.cuda.OutOfMemoryError
```
Fixes (try in order):
- Reduce `generation.max_new_tokens` to 64 or 128 in the config
- Use `dtype: float16` instead of `bfloat16`
- For tracing: reduce max tokens further — tracing with hooks is memory-heavy

### 3. OOM during tracing
Tracing stores activations at every generation step for every configured layer.
Fix: in `configs/experiments/tracing.yaml`, add `generation.max_new_tokens: 32`.
This is usually sufficient for observing the early-position dynamics.

### 4. Direction file not found when running tracing/patching
Fix: run `extract_refusal_direction.py` first.

### 5. Projection values all near zero
This would indicate the direction doesn't separate harmful/benign in activation space.
Possible causes:
- Too few held-out prompts (try increasing `held_out_n`)
- Wrong `direction_position` (try position 0 instead of -1)
- The last-token activations of the prompt are dominated by formatting tokens

### 6. Patching has no effect (no refusal_restored cases)
Possible interpretations:
- Refusal is committed before the patched position (try earlier target positions)
- The patched layer doesn't carry the causal signal (try more layers)
- The direction extraction needs refinement

This is a scientifically interesting null result — document it.

### 7. Tokenizer produces unexpected prefix lengths
```
WARNING: Prefix tokenizes to only N tokens, but k=K requested.
```
Check with:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
ids = tok.encode("Sure, here is how you can do that:", add_special_tokens=False)
print(len(ids), tok.convert_ids_to_tokens(ids))
```

### 8. Placeholder data in output
If you see `"source": "synthetic_placeholder"` in outputs, you forgot to replace
the fake prompts with real benchmark data. See `data/README.md`.

---

## Local vs. Cluster

### Local (dev / smoke-testing)
Use 3B to keep iteration fast. Limit `data.max_prompts: 10` in config for quick checks.

### Cluster (8B production runs)
- Request at least 24GB VRAM (A100-40GB or H100 recommended)
- Set `HF_HOME` to a shared filesystem path to avoid re-downloading the model
- Use SLURM array jobs if sweeping multiple configurations

---

## References

- Zou et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043
- Zou et al. (2023). "Representation Engineering." arXiv:2310.01405
- Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717
- Wei et al. (2023). "Jailbroken: How Does LLM Safety Training Fail?" NeurIPS 2023
- Taori et al. (2023). Alpaca: A Strong Open-Source Instruction-Following Model. Stanford CRFM.
