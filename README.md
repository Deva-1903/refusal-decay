# Refusal Decay

**Tracing the positional decay of the refusal direction in safety-aligned LLMs under prefilling attacks.**

COMPSCI 602 — Mechanistic Interpretability / Causal Analysis of LLM Safety Behavior.

---

## Project Goal

We study how the *refusal direction* — a feature direction in the residual stream that distinguishes harmful from benign inputs — evolves **token-by-token** during generation when the model is subjected to **prefilling attacks** (forced compliant prefixes).

The central question:

> Does the refusal signal at early token positions causally determine whether the model continues to refuse or comply at later positions? And does forcing an initial compliant token progressively erode this signal?

---

## Scope (Progress Report Stage)

This codebase implements the infrastructure and preliminary experiments for a progress report. The scope is intentionally narrow:

| Aspect | In scope | Out of scope (for now) |
|--------|----------|------------------------|
| Model | Llama-3.2-3B-Instruct | Larger models (until infra works) |
| Attack | Prefilling (k tokens) | Adversarial suffixes, GCG |
| Mechanism | Refusal direction (residual stream) | Attention heads, neurons |
| Probe | Difference-in-means | DAS, linear probes, nonlinear |
| Experiment | Tracing + pilot patching | Full causal ablation study |

---

## Repository Structure

```
refusal-decay/
├── README.md
├── RESEARCH_LOG.md             # tracks decisions, failures, scope changes
├── requirements.txt
├── environment.yml
├── .env.example
│
├── configs/
│   ├── base.yaml               # all defaults live here
│   ├── model_3b.yaml           # Llama-3.2-3B override
│   ├── model_8b.yaml           # Llama-3.1-8B override
│   └── experiments/
│       ├── baseline.yaml
│       ├── prefilling_sweep.yaml
│       ├── tracing.yaml
│       └── patching.yaml
│
├── data/
│   ├── harmful_prompts.jsonl   # 25 fake example harmful prompts
│   ├── benign_prompts.jsonl    # 25 fake example benign prompts
│   └── processed/              # normalized Prompt objects
│
├── src/
│   ├── config.py               # YAML loading + dot-access Config
│   ├── data/
│   │   ├── schema.py           # Prompt dataclass
│   │   └── loader.py           # JSONL loading + normalization
│   ├── generation/
│   │   ├── generator.py        # model loading + generation pipeline
│   │   └── prefilling.py       # prefilling attack implementation
│   ├── classification/
│   │   └── refusal_classifier.py  # phrase-list + Llama Guard stub
│   ├── probing/
│   │   ├── direction.py        # difference-in-means refusal direction
│   │   └── tracing.py          # per-token projection tracing
│   ├── patching/
│   │   └── patch.py            # pilot direction-patching experiment
│   └── utils/
│       ├── logging_utils.py
│       └── io_utils.py
│
├── scripts/
│   ├── run_baseline.py         # normal generation (k=0)
│   ├── run_prefilling_sweep.py # sweep k in {0,1,3,5,10}
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
├── outputs/                    # all generated artifacts (gitignored)
│   ├── generations/
│   ├── directions/
│   ├── traces/
│   ├── patches/
│   ├── plots/
│   └── logs/
│
├── notebooks/
│   └── exploration.ipynb
│
└── tests/
    ├── test_loader.py
    ├── test_config.py
    ├── test_direction.py
    └── test_schema.py
```

---

## Installation

### Local (Mac / Linux)

```bash
git clone <repo>
cd refusal-decay

# Conda (recommended)
conda env create -f environment.yml
conda activate refusal-decay

# Or pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your HuggingFace token (needed for Llama access)
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

### Cluster (SLURM)

```bash
# On login node
git clone <repo>
cd refusal-decay
conda env create -f environment.yml

# Set HF_TOKEN in your .bashrc or pass via sbatch --export
echo 'export HF_TOKEN=hf_...' >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start: Run Everything Locally

These commands assume you have the model and token set up.

```bash
# 1. Run tests (no GPU needed)
pytest tests/ -v

# 2. Baseline generation (k=0, ~5 min on 3B with one GPU)
python scripts/run_baseline.py

# 3. Prefilling sweep (k in {0,1,3,5,10}, ~25 min)
python scripts/run_prefilling_sweep.py

# 4. Extract refusal direction (~5 min)
python scripts/extract_refusal_direction.py

# 5. Run tracing (~30 min for 25 prompts × 5 k values)
python scripts/run_tracing.py

# 6. Run patching pilot (~20 min)
python scripts/run_patching.py

# 7. Generate all plots
python scripts/plot_results.py
```

All outputs go to `outputs/`. Each script is **resume-safe**: re-running will skip files that already exist (use `--no-resume` to force rerun).

---

## Swapping to the 8B Model

```bash
python scripts/run_baseline.py --model-config configs/model_8b.yaml
python scripts/run_prefilling_sweep.py --model-config configs/model_8b.yaml
python scripts/extract_refusal_direction.py --model-config configs/model_8b.yaml
# etc.
```

The 8B model requires ~20GB VRAM. Adjust SLURM `--mem` and `--gres` accordingly.

---

## Running on a SLURM Cluster

```bash
# Submit individual jobs
sbatch slurm/baseline.sh
sbatch slurm/tracing.sh
sbatch slurm/patching.sh

# Check status
squeue -u $USER

# View logs
tail -f outputs/logs/baseline_<JOB_ID>.out
```

**Before submitting**: edit the SLURM scripts to match your cluster:
- `--partition` — your GPU partition name
- `--gres=gpu:1` — may need `gpu:a100:1` or similar
- `conda activate refusal-decay` — adjust if using modules or venv

---

## Configuration System

All behavior is controlled by YAML configs. The base config (`configs/base.yaml`) sets all defaults; experiment configs override specific keys.

```bash
# Override k values from the command line
python scripts/run_prefilling_sweep.py --k-values 0 1 5 10

# Override held-out count for direction extraction
python scripts/extract_refusal_direction.py --held-out-n 30

# Use a different output directory by editing the experiment config:
# output:
#   generations_dir: outputs/my_experiment/
```

Config snapshots are saved alongside all outputs (as `config_snapshot.yaml`) for reproducibility.

---

## Output Format

### Generations (`outputs/generations/`)
JSONL files with one record per prompt:
```json
{
  "prompt_id": "h001",
  "label": "harmful",
  "category": "weapons",
  "prompt_text": "...",
  "prefix_k": 3,
  "generated_text": "...",
  "n_generated_tokens": 142,
  "refusal_label": "refusal",
  "matched_phrase": "I cannot"
}
```

### Traces (`outputs/traces/`)
Parquet/CSV files (tidy long format):
```
prompt_id | label | prefix_k | step | layer | projection
h001      | harmful | 0      | 1    | 8     | 0.342
h001      | harmful | 0      | 2    | 8     | 0.289
...
```

### Directions (`outputs/directions/`)
PyTorch `.pt` file: `{layer_idx: tensor(hidden_size)}`

### Patches (`outputs/patches/`)
JSONL files with baseline and patched generations + refusal labels.

---

## First 24 Hours Plan

Work in this order — each step validates the previous:

**Hour 1–2: Environment**
- [ ] Install deps, confirm `pytest tests/ -v` passes (no GPU needed)
- [ ] Set `HF_TOKEN` and confirm `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"`

**Hour 3–5: Baseline**
- [ ] Run `python scripts/run_baseline.py`
- [ ] Open `outputs/generations/baseline/k00.jsonl`, read 3–5 outputs manually
- [ ] Check `baseline_classified.jsonl` — does the refusal rate look ~right? (~90%+ expected)

**Hour 5–7: Prefilling sweep**
- [ ] Run `python scripts/run_prefilling_sweep.py`
- [ ] Read `outputs/generations/prefilling_sweep/refusal_rate_summary.csv`
- [ ] Does refusal rate drop as k increases? (Expected: yes, significantly)

**Hour 7–9: Direction extraction**
- [ ] Run `python scripts/extract_refusal_direction.py`
- [ ] Check the `.pt` file loads correctly: `python -c "from src.probing.direction import load_direction; d = load_direction('outputs/directions/refusal_direction.pt'); print(d[8].shape)"`
- [ ] Sanity check: does the direction norm equal 1.0?

**Hour 9–12: Tracing**
- [ ] Run `python scripts/run_tracing.py` (can limit to 5 prompts first with `data.max_prompts: 5`)
- [ ] Run `python scripts/plot_results.py --trace-dir outputs/traces`
- [ ] Open `outputs/plots/projection_vs_position_layer8.png` — does the projection decay with token position for k=0? Does it start lower for higher k?

**Hour 12–16: Patching**
- [ ] Run `python scripts/run_patching.py --config configs/experiments/patching.yaml`
- [ ] Check `outputs/patches/patching_classified.jsonl` — do any cases flip from refusal to compliance?

**Hour 16–24: Analysis + Writing**
- [ ] `python scripts/plot_results.py` — generate all 4 plot types
- [ ] Draft progress report sections: methods, preliminary results, figures

---

## Common Failure Modes

### 1. `HF_TOKEN` not set or invalid
```
OSError: You are trying to access a gated repo...
```
Fix: `export HF_TOKEN=hf_...` (get token from huggingface.co/settings/tokens with read access to Llama models).

### 2. CUDA OOM during generation
```
torch.cuda.OutOfMemoryError
```
Fixes:
- Reduce `generation.max_new_tokens` in config (try 64 or 128)
- Set `generation.batch_size: 1` (already default)
- Use `dtype: float16` instead of `bfloat16`
- Switch to smaller model if 8B is the issue

### 3. OOM during tracing (hooks store all activations)
Fix: Reduce `max_new_tokens` in tracing — tracing with 256 tokens × 8 layers × 25 prompts is expensive. Set `max_new_tokens: 32` in `configs/experiments/tracing.yaml` for initial exploration.

### 4. Direction file not found when running tracing/patching
```
FileNotFoundError: Refusal direction file not found
```
Fix: Run `python scripts/extract_refusal_direction.py` first.

### 5. Patching hook fires at wrong step
The step counter in `patch.py` increments per autoregressive step. If the model uses KV-caching (it does by default), the prefill step has `seq_len > 1` and subsequent steps have `seq_len = 1`. The hook correctly distinguishes these. If you see unexpected behavior, add `--log-level DEBUG` and check the hook fire messages.

### 6. Tokenizer produces unexpected prefix lengths
Some prefix strings tokenize differently than expected (e.g., "Sure" may be 2 tokens with a leading space). The `build_prefilled_input` function warns when fewer tokens are available than requested. Check with:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
ids = tok.encode("Sure, here is how you can do that:", add_special_tokens=False)
print(len(ids), ids)
```

### 7. Refusal classifier false positives/negatives
The phrase-list classifier is a heuristic. It may:
- **False positive**: classify "I cannot stress enough how important..." as refusal
- **False negative**: miss a refusal that uses unusual phrasing

For final results, validate a sample manually. Consider setting `prefix_chars` higher or lower based on what you observe.

### 8. SLURM job fails immediately
- Check `outputs/logs/baseline_<JOB_ID>.err`
- Common causes: wrong partition name, conda not found, `HF_TOKEN` not exported to job env
- Fix HF_TOKEN: add `--export=ALL` to sbatch, or `export HF_TOKEN=...` in the script

---

## How to Run: Local vs. Cluster

### Local (development / debugging)

```bash
# Limit prompts for fast iteration
# Edit configs/base.yaml:  data.max_prompts: 5

python scripts/run_baseline.py
```

Local GPU (single A100 40GB or similar): the 3B model fits comfortably. The 8B model needs ~20GB.

### Cluster (full experiments)

1. SSH to login node, clone repo, create conda env
2. Edit SLURM scripts to match cluster partition/memory
3. Submit jobs:
```bash
# Run in order (direction must precede tracing/patching)
sbatch slurm/baseline.sh
# Wait for completion, then:
JID=$(sbatch --parsable slurm/tracing.sh)
# patching can run in parallel with tracing
sbatch slurm/patching.sh
```
4. Outputs are in `outputs/` — copy back with `rsync` or read via mounted filesystem.

**Tip**: For quick iteration on the cluster, use an interactive session:
```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
conda activate refusal-decay
python scripts/run_baseline.py
```

---

## Adding Real Datasets

Replace `data/harmful_prompts.jsonl` with a real harmful prompt dataset:
- **AdvBench**: Zou et al. 2023 — behavioral test set of 500 harmful requests
- **HarmBench**: Mazeika et al. 2024 — standardized harm evaluation benchmark

Schema: each line must have `{"prompt_id": str, "text": str, "category": str, "source": str}`.

The pipeline will automatically detect and use it.

---

## TODO

- [ ] Replace fake example data with real AdvBench / HarmBench datasets
- [ ] Validate refusal classifier against manual annotations
- [ ] Add Llama Guard integration in `classification/refusal_classifier.py`
- [ ] Extend patching to cross-prompt (source prompt → target prompt)
- [ ] Add attention-head localization as a future module
- [ ] Confirm correct handling of `device_map="auto"` with multi-GPU hooks
- [ ] Add experiment tracking (wandb or simple CSV log)
- [ ] Profile memory usage during tracing for large k sweeps

---

## Citation / References

- Zou et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." [arXiv:2307.15043]
- Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." [arXiv:2310.01405]
- Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." [arXiv:2406.11717]
- Wei et al. (2024). "Jailbroken: How Does LLM Safety Training Fail?" [NeurIPS 2023]
- Qi et al. (2023). "Fine-tuning Aligned Language Models Compromises Safety." [arXiv:2310.03693]
