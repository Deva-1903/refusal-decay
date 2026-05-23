# Paper Plan — arXiv preprint

Target: a credible arXiv preprint (cs.CL / cs.LG) that a knowledgeable
interpretability/safety reader finds defensible. Built on the COMPSCI 602
project, but raised to research-publication rigor.

---

## 1. The thesis (sharper than the course report)

> In the clean setting, the refusal direction causally controls refusal
> (ablating it makes an aligned model comply; adding it makes it refuse).
> Under a prefilling attack, the **same** direction-level intervention that
> works clean **fails to restore refusal**, even though the late-layer
> refusal-direction signal is still present, depth-localized, and
> prompt-level predictive. Prefilling therefore defeats refusal through a
> mechanism the single refusal direction does not capture at the residual
> stream.

The contribution is the **contrast**: direction is causal clean, not
recoverable under attack. That contrast is what makes the negative result a
finding rather than a failed experiment.

---

## 2. Why the current results are not yet publishable

| Gap | Why a reviewer rejects it | Fix |
|-----|---------------------------|-----|
| Additive intervention had **zero effect even on benign prompts** | Reads as an underpowered edit, not specificity. No proof the method can move behavior at all, so the null is uninterpretable. | **Positive control**: Arditi-style all-position ablation/addition that demonstrably moves behavior in the clean setting. |
| **n = 25** | CIs too wide; single-prompt noise dominates. | Scale to **≥150** harmful + matched benign. |
| **Single model** | Can't tell if the late-layer gradient is Llama-specific. | Add **Qwen2.5-7B-Instruct** and **Llama-3.2-3B** (≥1 cross-family). |
| **Single-position, single-layer edit** | "Maybe it's just too local." | **All-position, multi-layer sweep** + α sweep. |
| No mechanistic depth beyond projection | "You measured a readout, not a mechanism." | One slice: **logit-lens / direct-effect** on the refusal token, or attention-pattern shift. |

---

## 3. Experiments (priority order)

Each maps to a fix above. P0 items are required; P1 strengthen.

### E1 (P0) — Positive control: directional intervention works in the clean setting
The linchpin. Implement Arditi-style **directional ablation** (project the
direction out of the residual at **every token position, every layer**) and
**directional addition** (add `α·d̂` at every position/layer).
- Clean harmful prompts (k=0): ablation should **drop** refusal; this proves
  the direction is causal and the method works.
- Clean benign prompts (k=0): addition should **induce** refusal (the Arditi
  effect). This is the positive control that was missing.
- Expected: large behavioral movement. If we don't see it, the intervention
  implementation is wrong and must be fixed before any null is reported.

### E2 (P0) — The headline contrast: same intervention under prefilling
Run the **identical** all-position ablation/addition on attacked (k=3, k=10)
harmful prompts.
- Hypothesis: adding the direction back does **not** restore refusal under
  attack, in sharp contrast to E1.
- This is the paper's central result.

### E3 (P0) — Scale + behavioral/observational replication
- ≥150 harmful (AdvBench) + matched benign (Alpaca), k ∈ {0, 3, 10}.
- Re-confirm: refusal-rate drop, late-layer monotone-by-depth negative shift
  (held-out direction), prompt-level association. With proper n and CIs.

### E4 (P1) — Cross-model generality
- Repeat E1–E3 on **Qwen2.5-7B-Instruct** (cross-family) and
  **Llama-3.2-3B** (same family, smaller).
- Claim becomes "holds across N models" or, if it doesn't, "Llama-specific" —
  either is publishable if reported honestly.

### E5 (P1) — One mechanistic slice
- Logit-lens / direct-effect: project late-layer residual onto the
  unembedding and measure the direct contribution to the first-token refusal
  vs. compliance logits, clean vs. attacked.
- Goal: a sentence about *where* the attack acts that the projection trace
  alone can't support.

### Carried over from the course project (already done, just re-run at scale)
- Difference-in-means held-out direction extraction.
- Cross-condition patching (now a secondary result, framed as "even a clean
  source state doesn't recover it").
- Multi-seed random/orthogonal controls; bootstrap CIs; McNemar; classifier
  spot-check.

---

## 4. Code to build

| Component | Status | File |
|-----------|--------|------|
| All-position directional ablation + addition (every layer/position) | **NEW** | `src/patching/directional_intervention.py` |
| Positive-control runner (clean ablate/add, harmful + benign) | **NEW** | `scripts/run_directional_intervention.py` |
| Configs: positive control, attacked contrast | **NEW** | `configs/experiments/paper/*.yaml` |
| Model configs: Qwen2.5-7B, Gemma-2-9B | **NEW** | `configs/model_qwen7b.yaml`, `configs/model_gemma9b.yaml` |
| Scale-n data + config | **NEW** | larger prompt slices |
| Logit-lens / direct-effect | **NEW** | `src/probing/logit_lens.py` |
| Direction extraction, tracing, generation, stats | reuse | existing `scripts/` |

Most observational + stats infrastructure already exists from the course
project. The genuinely new code is the **all-position directional
intervention** (E1/E2) and the **logit-lens** (E5).

---

## 5. Unity run plan (high level)

Per model: extract held-out direction → behavioral generation (k=0/3/10) →
tracing → E1 clean ablate/add → E2 attacked ablate/add → stats.
- 8B model: ~A100/L40S, a few hours per full pass.
- Budget ~1-2 days wall-clock across 2-3 models with queueing.
- A `slurm/paper_pipeline.sh` will orchestrate one model end to end;
  parametrize by `--model-config`.

---

## 6. Paper outline (arXiv, ~6-8 pages)

1. **Abstract** — direction is causal clean, not recoverable under prefilling.
2. **Introduction** — shallow alignment (Qi), refusal direction (Arditi), the
   gap: does the direction stay causal under attack?
3. **Background / Related Work** — refusal direction, prefilling/shallow
   alignment, activation patching, safety-component interp (Chen, Zhou).
4. **Method** — STEP setup; difference-in-means direction (held-out);
   directional ablation/addition; cross-condition patching; controls; stats.
5. **Results**
   - 5.1 Behavioral attack effect + scale.
   - 5.2 Observational: late-layer depth gradient + prompt-level association.
   - 5.3 **Positive control (E1): the direction is causal clean.**
   - 5.4 **Headline (E2): the same intervention fails under prefilling.**
   - 5.5 Cross-model (E4).
   - 5.6 Mechanistic slice (E5).
6. **Discussion** — what defeats refusal if not the direction; bounded claims.
7. **Limitations & threats to validity** — honest, thorough.
8. **Conclusion.**

---

## 7. Honest risk register

- **E1 might show the intervention barely moves behavior** even clean (if our
  direction or hook is weak). If so, the whole paper is blocked until fixed —
  which is exactly why E1 is P0 and runs first.
- **E2 might actually show partial recovery** with the stronger all-position
  intervention. If it does, the thesis flips to "information-loss," which is
  *also* a clean, publishable result — we follow the evidence.
- **Cross-model (E4) might not replicate.** Then the claim narrows to Llama;
  still fine, just scoped.
- Negative-result papers are accepted at workshops but need airtight controls.
  The positive control (E1) is the difference between "rigorous null" and
  "we couldn't get it to work."

---

## 8. Sequencing

1. Build E1/E2 intervention code + configs (this repo, now).
2. Smoke-test on 3B locally / cheap GPU.
3. Run E1 on 8B first — **gate**: confirm the intervention moves behavior
   clean before investing in the rest.
4. If gate passes: E2, E3, then E4/E5.
5. Analyze, then draft the paper into `paper/`.
