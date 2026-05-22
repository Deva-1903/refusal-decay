# Refusal Decay

**Tracing refusal-direction changes under prefilling attacks in safety-aligned LLMs.**

A mechanistic-interpretability study of *why* prefilling attacks defeat safety
alignment in Llama-3.1-8B-Instruct, and whether the internal "refusal signal"
they disrupt can be intervened on to recover refusal behavior.

> Final course project for **COMPSCI 602** (Research Methods in Computer Science),
> UMass Amherst, Spring 2026.

---

## 📄 Reports

The full write-up lives in [`reports/`](reports/). Start with the final paper:

| Report | What it covers |
|--------|----------------|
| **[08: Final Paper](reports/08-final-paper.pdf)** | **Complete research-paper synthesis. Read this one.** |
| [07: Revised Results](reports/07-revised-results.pdf) | Held-out direction, cross-condition patching, additive intervention, controls |
| [06: Initial Results](reports/06-initial-results.pdf) | First execution of the research design |
| [05: Research Design](reports/05-research-design.pdf) | Experimental plan, hypotheses, analysis methods |
| [03: Exploratory Results](reports/03-exploratory-results.pdf) | First contact with the system; the surprising late-layer finding |

LaTeX sources for the later reports are under [`reports/latex/`](reports/latex/).

---

## TL;DR: what we found

A **prefilling attack** forces a chat model to begin its reply with a compliant
prefix (e.g. *"Sure, here is how you can do that:"*). This reliably breaks
refusal. We measured what happens inside the model using the difference-in-means
**refusal direction** [(Arditi et al., 2024)](https://arxiv.org/abs/2406.11717).

**Observational findings (all hold under a held-out direction):**
- Prefilling at `k=3` drops refusal from **0.92 → 0.32**.
- The late-layer (24, 27) refusal-direction projection shifts **strongly
  negative** under attack, with a **monotone-by-depth gradient**
  (Δ = 0.76, 1.63, 3.23, 4.08 at layers 16/20/24/27).
- Within an attacked batch, prompts that *still refuse* have less-negative
  late-layer projection than prompts that comply; the same depth signature at
  the prompt level.

**Causal findings (the twist):**
- Patching a clean `k=0` refusal-direction component into the attacked forward
  pass **does not restore refusal** (every cell ≤ 0.08, 95% CIs include zero).
- Directly adding a scaled refusal-direction vector restores refusal in
  **zero of 300** late-layer prompts, statistically indistinguishable from
  random and orthogonal control vectors (5 seeds each).
- A **benign positive control** produces zero false-refusals, ruling out a
  "the direction is a free knob" explanation.

**Interpretation:** the late-layer refusal-direction signal is a *real and
prompt-level-predictive readout* of the attacked state, but at the intervention
granularity tested it is **not the causal lever** for refusal. The attack
appears to lock in compliance earlier (during the prefill) or via downstream
computation that single-position residual edits cannot overturn. This conclusion
is bounded explicitly to the interventions tested.

---

## STEP framing

- **System**: Llama-3.1-8B-Instruct (32 layers, hidden 4096), bfloat16.
- **Task**: safety-constrained generation: refuse harmful prompts, comply with benign ones.
- **Environment**: prefilling attacks with `k ∈ {0, 3, 10}` forced compliant tokens.
- **Phenomenon**: joint behavioral refusal failure + late-layer refusal-direction shift.

Harmful prompts from [AdvBench](https://arxiv.org/abs/2307.15043); benign
controls from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

---

## Repository structure

```
reports/            Compiled report PDFs (+ LaTeX source under reports/latex/)
src/                Core library: config, data, generation, probing, patching, classification
scripts/            Runnable experiment + analysis scripts
configs/            YAML experiment configs (base + per-report overrides)
data/               Harmful (AdvBench) and benign (Alpaca) prompt sets
slurm/              SLURM batch scripts for HPC (UMass Unity)
outputs/            Summary CSVs + plots for reports 6 and 7 (bulky raw dumps trimmed)
tests/              Unit tests
```

---

## Reproducing the headline pipeline

Set up the environment (conda or pip):

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate refusal-decay
export HF_TOKEN=<your Hugging Face token>  # Llama-3.1-8B is gated
```

Run the Report 7 pipeline (held-out direction → tracing → cross-condition
patching → additive intervention → benign control → stats → plots):

```bash
# On a single A100/L40S-class GPU
sbatch slurm/report7_pipeline.sh
# or run the individual scripts in scripts/run_report7_*.py
```

Methodology notes:
- Refusal direction = difference-in-means of last-token activations on a
  **held-out** 50 harmful + 50 benign set, disjoint from the traced prompts.
- Intervention restoration/loss rates reported with **bootstrap 95% CIs** and
  **McNemar's exact test**.
- Direction-specificity controls: random and orthogonal unit vectors (5 seeds);
  benign positive control for false-refusal induction.

---

## Scope

| Aspect | In scope | Out of scope (future work) |
|--------|----------|----------------------------|
| Model | Llama-3.1-8B-Instruct | Other / larger model families |
| Attack | Prefilling (`k` forced tokens) | Adversarial suffixes, GCG |
| Mechanism | Refusal direction (residual stream) | Attention heads, neurons |
| Intervention | Single-position residual edits | Multi-position / attention-routing edits |

The causal conclusion is deliberately bounded to the single-position
intervention granularity tested; broader intervention sites are the natural
next step.

---

## References

- Arditi et al. (2024), *Refusal in Language Models is Mediated by a Single Direction.* arXiv:2406.11717
- Qi et al. (2025), *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR
- Zou et al. (2023), *Universal and Transferable Adversarial Attacks on Aligned Language Models* (AdvBench). arXiv:2307.15043
- Chen et al. (2025), *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons.* NeurIPS
- Zhou et al. (2025), *On the Role of Attention Heads in Large Language Model Safety.* ICLR

---

*This is defensive AI-safety research: understanding why jailbreak-style attacks
succeed in order to inform more robust alignment.*
