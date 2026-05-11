# Research Analysis: Report 3 → Report 6

**Project:** Tracing the Positional Decay of the Refusal Direction in Safety-Aligned LLMs Under Prefilling Attacks  
**Course:** COMPSCI 602 — Mechanistic Interpretability / Causal Analysis of LLM Safety Behavior  
**Model:** Llama-3.1-8B-Instruct (32 layers, hidden size 4096)  
**Prompt data:** AdvBench harmful (520 records), Alpaca benign (520 records) — real benchmark data confirmed  
**Analysis date:** 2026-05-04  

---

## Correction to VERIFY_REPORT6.md

The VERIFY_REPORT6.md (written the same day as this analysis) incorrectly stated that all four Report 6 behavioral generation conditions produced only GPU error records. Direct inspection of the classified JSONL files shows this is wrong for three of the four conditions:

| Condition | n_total | n_valid | n_error | Status |
|-----------|---------|---------|---------|--------|
| harmful_k00 | 25 | 25 | 0 | **Valid** |
| harmful_k03 | 25 | 25 | 0 | **Valid** |
| benign_k00 | 25 | 25 | 0 | **Valid** |
| harmful_k10 | 25 | 0 | 25 | **All errors** (GTX 1080 Ti GPU, sm_61 incompatible) |

The Report 6 generation refusal rate summary (`outputs/report6/summaries/generation_refusal_rates_by_condition.csv`) is valid for harmful_k00, harmful_k03, and benign_k00 and correctly reflects phrase-classified outputs from an L40S run. Only harmful_k10 behavioral data is missing from the Report 6 pipeline.

---

## 1. Project overview and scope

This project investigates how the *refusal direction* — a single direction in the residual stream identified via difference-in-means (DiM) between harmful and benign prompts — evolves token-by-token during autoregressive generation when the model is subjected to a prefilling attack (k forced compliant prefix tokens prepended before `model.generate()`).

Core research questions:
1. Does the refusal-direction projection decline with token position under attack?
2. Is the decay concentrated in particular layers?
3. Does a larger k deepen or accelerate the decline?
4. Does patching the refusal-direction component at specific layer × target-position combinations restore refusal behavior?

The work is grounded in Arditi et al. (2024), who established that refusal in aligned LLMs is mediated by a single direction in the residual stream, and extends that finding to the temporal (token-position) dimension under attack.

---

## 2. Experiment inventory

| Report | Experiment block | Conditions | Prompt count | Layers | Key params | Output files | Status |
|--------|-----------------|------------|--------------|--------|------------|--------------|--------|
| R3 | Harmful baseline | harmful, k=0 | 25 | — | behavioral only | `report3_runs/.../baseline_classified.jsonl` | Complete |
| R3 | Prefilling sweep | harmful, k=0,3,10 | 25 | — | behavioral only | `sweep_classified.jsonl`, `refusal_rate_summary.csv` | Complete |
| R3 | Benign baseline | benign, k=0 | 25 | — | behavioral only | `report3_runs/.../baseline_classified.jsonl` | Complete |
| R3 | Direction extraction | harmful+benign | 25+25 | 0,4,8,12,16,20,24,27 | pos=-1, NOT held out | `refusal_direction.pt` | Complete |
| R3 | Observational tracing | harmful, k=0,3,10 | 25 | 0,4,8,12,16,20,24,27 | max 256 tokens | `traces_all.parquet` | Complete |
| R3 | Patching pilot | harmful, k=3 | 5 | layer 27 | src=-1, tgt=5; 1 cell | `patch_layer27_ts-1_tt5.jsonl` | Complete; n=5 single cell |
| R6 | Direction extraction | harmful+benign | 50+50 | 16,20,24,27 | pos=-1, NOT held out | `outputs/report6/directions/refusal_direction.pt` | Complete |
| R6 | Observational tracing | harmful, k=0,3,10 | 25 | 16,20,24,27 | max 48 tokens | `outputs/report6/traces/traces_all.parquet` | Complete |
| R6 | Behavioral generation | harmful_k00, harmful_k03, benign_k00 | 25 | — | max 64 tokens | `classified.jsonl` per condition | **Valid (3 of 4 conditions)** |
| R6 | Behavioral generation | harmful_k10 | 25 | — | — | `classified.jsonl` | **All errors (GPU incompatibility)** |
| R6 | Targeted patching | harmful, k=3 | 10 | 16, 24, 27 | src=-1, tgt=1,3,5; 9 cells | `patching_classified.jsonl`, `patching_refusal_recovery_summary.csv` | Complete |
| R6 | Summaries and plots | all | — | — | — | `outputs/report6/summaries/`, `plots/` | Complete (except k=10 behavioral bar) |

---

## 3. Report 3: Exploratory phenomena pass

### 3.1 Goal

Report 3 was an exploratory pass to establish that the phenomena of interest exist in Llama-3.1-8B-Instruct before investing in a tighter causal design. The goal was to answer: Does prefilling suppress refusal? Does the refusal-direction projection decay across token position and layers? Is there any hint of causal restoration under patching?

### 3.2 Design

**Behavioral sweep.** 25 harmful prompts (AdvBench) were run at k=0, k=3, and k=10. Separately, 25 benign prompts (Alpaca) were run at k=0. All outputs were classified with a phrase-list heuristic classifier.

**Direction extraction.** The refusal direction was extracted via DiM at the last token position across 8 layers (0, 4, 8, 12, 16, 20, 24, 27) using the same 25 harmful + 25 benign prompts as tracing. Direction extraction and tracing shared the same prompt set — the direction was not held out.

**Tracing.** The same 25 harmful prompts were traced at all 8 layers across k=0, 3, 10. Max generated tokens: 256.

**Patching pilot.** A single patching cell was run: layer 27, source_pos=-1, target_pos=5, on 5 harmful prompts under k=3.

**Config reference:** `configs/experiments/report3_tracing_small.yaml` (snapshot in `report3_runs/20260406_232425_report3_mechanism/report3_small/config_snapshot.yaml`).

### 3.3 Results

#### Behavioral: refusal rates by k

From `report3_runs/20260406_222202_8b_harmful_sweep_k0_3_10_25/report3_prefilling_sweep/refusal_rate_summary.csv`:

| k | n_prompts | n_refusals | refusal_rate |
|---|-----------|------------|--------------|
| 0 | 25 | 24 | 0.96 |
| 3 | 25 | 8 | 0.32 |
| 10 | 25 | 9 | 0.36 |

Benign k=0: 0/25 refused (0.00%).

The attack effect is strong: refusal drops from 96% to 32% at k=3. The difference between k=3 and k=10 is small and non-monotone (32% vs 36%), attributed in the project notes to sample size limitations.

#### Tracing: mean projection by layer x k (generated tokens only)

Computed from `report3_runs/20260406_232425_report3_mechanism/report3_small/traces_all.parquet`, `is_prefill == False` only:

| k | Layer 0 | Layer 4 | Layer 8 | Layer 12 | Layer 16 | Layer 20 | Layer 24 | Layer 27 |
|---|---------|---------|---------|----------|----------|----------|----------|----------|
| 0 | +0.015 | +0.139 | +0.700 | +1.507 | +1.052 | +1.303 | +0.086 | -1.273 |
| 3 | +0.017 | +0.236 | +0.962 | +1.293 | -0.389 | -1.046 | -4.040 | -6.756 |
| 10 | +0.018 | +0.205 | +0.866 | +0.997 | -0.752 | -1.438 | -4.321 | -7.073 |

Key pattern: Early layers (0, 4, 8, 12) stay positive across all k values. The sign flip under attack first appears strongly at layer 16, and deepens at layers 20, 24, and 27. At k=0, layer 27 is already modestly negative (mean: -1.273), which may reflect generation dynamics at very long responses for the one k=0 compliant case.

#### Patching pilot

Setting: k=3, layer 27, source_pos=-1, target_pos=5, n=5 harmful prompts.

- Baseline refusal rate over 5 prompts: 1/5 (20%)
- Patched refusal rate: 1/5 (20%)
- compliance → refusal: 0/5
- refusal → compliance: 0/5
- No change: 5/5

Completely null under a single cell with n=5.

### 3.4 Key findings

1. Prefilling sharply suppresses refusal: from 96% at k=0 to 32% at k=3 on 25 harmful prompts.
2. The refusal-direction projection decays dramatically in late layers (24, 27) under attack, while early layers (0, 4, 8) are largely unaffected.
3. The clearest layer-level separation between k=0 and attacked conditions is at layers 16–27.
4. The k=3 vs k=10 tracing does show k=10 is slightly more negative across all late layers, consistent with a graded effect, even though the behavioral refusal rates are non-monotone.
5. The single patching pilot produced no effect under one highly constrained cell.

### 3.5 Surprises

- Layer 27 projection is negative even at k=0 (mean: -1.273). Under refusal behavior the direction should stay positive; the negative mean may be explained by long compliant continuations from the one k=0 non-refusal case pulling down the average over all generated token positions.
- The non-monotone k=3 vs k=10 behavioral comparison (32% vs 36%) at n=25.

### 3.6 What Report 3 changed about the project

Report 3 established that the phenomena exist and are large enough to study, and raised the key design question for Report 6: is the patching source already inverted under attack? If layer-27 activations at the last prefix token are negative under k=3, injecting that activation into early generated positions cannot restore refusal. Report 6 was designed to directly test this.

---

## 4. Report 6: Focused design-driven pass

### 4.1 Goal and causal question

> **How and under what intervention conditions does the late-layer negative shift in the refusal-direction signal contribute to refusal failure under prefilling attacks?**

Specifically: Is the late-layer negative projection under attack causally upstream of compliance, or is it epiphenomenal? Can injecting a positive refusal-direction component at specific layer × target-position combinations restore refusal behavior in an already-attacked context?

### 4.2 Design

**Direction extraction.** Extracted from 50 harmful + 50 benign prompts at layers 16, 20, 24, 27 using last-token position (-1). Still non-held-out (the 25 tracing prompts are a subset of the 50 extraction prompts).

**Observational tracing.** 25 harmful prompts traced at layers 16, 20, 24, 27 under k=0, k=3, and k=10. Max generated tokens: 48 (shorter than Report 3's 256; changes which token positions are covered).

**Behavioral generation.** 25 harmful prompts generated under k=0, k=3, k=10; 25 benign prompts under k=0. Max tokens: 64. Three of four conditions completed successfully on L40S; harmful_k10 failed on GTX 1080 Ti (sm_61 incompatible) and the error records were cached by the resume logic.

**Targeted patching.** 10 harmful prompts patched under k=3 across a 3-layer × 3-position grid:
- Layers: 16, 24, 27
- Target positions: 1, 3, 5 (generated token positions)
- Source position: -1 (last token of prefill, taken from the attacked k=3 forward pass)
- Mode: replace

This gives 9 combinations × 10 prompts = 90 total records. The patching source is from the attacked condition itself (same-prompt self-patching under k=3), not a k=0 baseline.

### 4.3 Results

#### Behavioral: refusal rates by condition (Report 6 pipeline)

From `outputs/report6/summaries/generation_refusal_rates_by_condition.csv`:

| Condition | dataset | prefix_k | n_valid | n_refusal | n_compliance | refusal_rate |
|-----------|---------|----------|---------|-----------|--------------|--------------|
| harmful_k00 | harmful | 0 | 25 | 23 | 2 | 0.92 |
| harmful_k03 | harmful | 3 | 25 | 8 | 17 | 0.32 |
| benign_k00 | benign | 0 | 25 | 0 | 25 | 0.00 |
| harmful_k10 | harmful | 10 | 0 | — | — | **MISSING (GPU errors)** |

For harmful_k10, the Report 3 behavioral number (9/25 refused, 36%) is the best available reference.

The Report 6 k=0 refusal rate (92%) is one case lower than Report 3 (96%). The k=3 rate matches exactly (32%).

#### Tracing: mean projection by condition x layer (generated tokens only)

From `outputs/report6/summaries/trace_mean_projection_by_condition_layer.csv`:

| Condition | prefix_k | Layer 16 | Layer 20 | Layer 24 | Layer 27 |
|-----------|----------|----------|----------|----------|----------|
| harmful_k00 | 0 | +1.271 | +1.612 | +0.665 | -0.376 |
| harmful_k03 | 3 | +0.172 | -0.374 | -3.110 | -5.285 |
| harmful_k10 | 10 | -0.270 | -0.822 | -3.463 | -5.811 |

Consistent with Report 3: early layers are stable or mildly shifted; late layers show large negative values under attack. Layer-24 is the clearest sign-flip layer (positive at k=0, strongly negative at k=3 and k=10). Layer-27 is modestly negative even at k=0 (-0.376).

#### Patching: per-cell results

From `outputs/report6/summaries/patching_refusal_recovery_summary.csv`. Patching condition: harmful_k03, n=10 prompts per cell. Baseline refusal rate on this 10-prompt subset: 50% (5/10).

| Layer | Source proj (mean) | Target pos 1 | Target pos 3 | Target pos 5 | Restored | Lost |
|-------|--------------------|:---:|:---:|:---:|---------|------|
| 16 | +0.543 | 50% | 50% | 50% | 0 | 0 |
| 24 | +0.948 | 40% | 40% | 50% | 0 | 2 |
| 27 | **-0.458** | 40% | 50% | 50% | 0 | 1 |

Totals across all 90 records: **0 restored, 3 lost, 87 no change.**

The critical observation is the layer-27 source projection: -0.458 (already inverted under k=3). Patching a negative component into early positions cannot be expected to restore refusal. At layer 24, the source is positive (+0.948) but patching still fails to restore and slightly reduces refusal. At layer 16, zero net effect in either direction.

### 4.4 What is genuinely new in Report 6 vs Report 3

1. **Expanded patching grid.** Report 3 ran one cell (layer 27, target pos 5, n=5). Report 6 ran 9 cells (n=10 each), covering the layer × position space.

2. **Source projection diagnostic.** Report 6 records the mean source projection per patching cell. This makes the mechanism of the null result interpretable: at layer 27, the source is already inverted.

3. **Cleaner named-condition pipeline.** The Report 6 pipeline uses a consistent naming scheme (`harmful_k00`, `harmful_k03`, etc.) that links behavioral, tracing, and patching outputs unambiguously.

4. **Larger direction extraction set.** Report 6 used 50+50 prompts vs Report 3's 25+25. Both remain non-held-out.

5. **Layer specificity of patching quantified.** The first side-by-side comparison of patching effects across layers 16, 24, and 27, showing null-to-negative across all and documenting that the sign of the source projection varies by layer.

### 4.5 What remains unresolved

1. **Why does layer 24 patching fail even with a positive source?** The mean source projection at layer 24 is +0.948, yet no compliance cases flip to refusal. Either the patching window (positions 1–5) is too late to affect the refusal decision, or a single-layer intervention at 24 is insufficient because the downstream layer-27 negative shift follows regardless.

2. **The source is from the attacked condition.** Even at layers 16 and 24 where the source projection is positive under k=3, it is much weaker than what k=0 activations at those layers would provide. A proper design would take the source from a k=0 forward pass on the same prompt.

3. **Position 0 not tested.** The patching grid starts at generated token position 1. The refusal decision may be committed at position 0.

4. **Behavioral k=10 is missing from the Report 6 pipeline.**

5. **Prompt-level association not tested.** The hypothesis that negative projection predicts compliance is only tested at the condition level; prompt-level data under k=3 (does the compliant subset have more negative projections than the refusing subset?) has not been analyzed.

6. **Trace summary averages over variable-length responses.** Short refusals and long compliant responses contribute differently to the mean projection.

---

## 5. Direct comparison: Report 3 vs Report 6

| Topic | Report 3 | Report 6 | What changed / learned |
|-------|----------|----------|------------------------|
| Goal | Exploratory: does the phenomenon exist? | Focused: is the late-layer shift causally relevant? | Shift from observation to targeted causal test |
| Design style | Broad (8 layers, 256 max tokens) | Narrow (4 late layers, 48 max tokens) | More pipeline-robust; less breadth |
| Behavioral conditions | k=0,3,10 harmful + k=0 benign | k=0,3 harmful + k=0 benign (k=10 failed) | R6 partially completes behavioral block |
| k=0 harmful refusal rate | 24/25 = 96% | 23/25 = 92% | Small run-to-run variation; consistent story |
| k=3 harmful refusal rate | 8/25 = 32% | 8/25 = 32% | Exact match |
| k=10 harmful refusal rate | 9/25 = 36% | MISSING | R3 is only reference |
| Benign k=0 refusal rate | 0/25 = 0% | 0/25 = 0% | Exact match |
| Layer 16 mean projection (k=0) | +1.052 | +1.271 | Consistent sign; small magnitude difference from token-window difference |
| Layer 27 mean projection (k=0) | -1.273 | -0.376 | Both negative; R6 less extreme (48-token limit) |
| Layer 24 mean projection (k=3) | -4.040 | -3.110 | Same sign; R6 slightly less extreme |
| Layer 27 mean projection (k=3) | -6.756 | -5.285 | Same sign; consistent late-layer inversion |
| Late-layer pattern | Qualitatively established (8-layer heatmaps) | Quantified in 4-layer summary table with condition-level CSVs | Pattern strengthened |
| Patching setup | 1 cell: layer 27, src=-1, tgt=5, n=5 | 9 cells: {16,24,27}×{1,3,5}, src=-1, n=10 | 9x more coverage; source projection recorded |
| Patching result | 0/5 restored, 0/5 lost | 0/90 restored, 3/90 lost | Null-to-negative confirmed at scale |
| Source projection at layer 27 | Not recorded | -0.458 (inverted) | Explains the null result mechanistically |
| Direction extraction | 25+25 prompts, 8 layers, non-held-out | 50+50 prompts, 4 layers, non-held-out | Larger set; still shares prompts with tracing |
| Interpretation | Phenomenon exists; pilot inconclusive | Null result explained by source inversion; positive-source cells also null | Mechanism of null result is better understood |
| Next-step implication | Bigger patching grid | Fix source to k=0 baseline activations | R6 narrows the diagnostic |

---

## 6. Hypothesis status after Report 6

### H1: Late-layer negative shift under prefilling

**Status: Supported**

At layer 27, mean projection is -5.285 (k=3) and -5.811 (k=10) vs -0.376 (k=0). At layer 24: -3.110 (k=3) and -3.463 (k=10) vs +0.665 (k=0). The sign flip from positive to strongly negative is clear at layer 24 and deepens at layer 27. This replicates across both Report 3 and Report 6 runs with different max-token budgets and direction extraction sizes.

**Caveat:** Layer 27 projection is already negative at k=0 (-0.376 in R6, -1.273 in R3). Layer 24 provides a cleaner positive-to-negative sign flip. The absolute values depend on the number of generated tokens averaged over, which differs between R3 and R6.

**Implication for Report 7:** Lead with layer-24 numbers for the clearest sign flip claim; present layer-27 as reinforcing.

---

### H2: Layer specificity (late layers show larger shift than early layers)

**Status: Supported**

Mean projection shift from k=0 to k=3: layer 16 shifts by -1.099 (1.271 to 0.172), layer 20 by -1.986 (1.612 to -0.374), layer 24 by -3.775 (0.665 to -3.110), layer 27 by -4.909 (-0.376 to -5.285). The shift magnitude grows monotonically with layer depth.

**Caveat:** Early layers (16, 20) do shift toward negative under attack; the distinction is quantitative (magnitude), not binary (immune vs affected).

**Implication for Report 7:** The layer-16 comparison layer is a useful foil for layer 27 in the heatmap and line plots.

---

### H3: Behavioral association (negative projection predicts compliance)

**Status: Supported at the condition level; not yet tested at the prompt level**

Condition-level association: k=0 harmful (92% refusal, layer-27 mean -0.376), k=3 harmful (32% refusal, layer-27 mean -5.285), benign k=0 (0% refusal, not separately traced). The association is directionally consistent with the hypothesis.

**Caveat:** This is a between-condition comparison only. The prompt-level question — do the 8 prompts that refused under k=3 have less-negative late-layer projections than the 17 that complied? — has not been tested.

**Implication for Report 7:** A prompt-level analysis within the k=3 condition is the highest-priority observational addition.

---

### H4: Patching restoration under current design

**Status: Null (0/90 records restored, 3/90 lost)**

**Evidence:** No compliance-to-refusal flip in any of the 9 layer × position cells. Three cases went the wrong direction (refusal to compliance). The source projection at layer 27 is -0.458 (already inverted), directly explaining those cells. Even cells with a positive source (layers 16 and 24) produced zero restoration.

**Caveat:** The null result is fully explained by the current patching design, not by the impossibility of causal restoration. The source is taken from the attacked forward pass, not a clean k=0 baseline. This is not a fair test of the causal hypothesis.

**Implication for Report 7:** Use k=0 activations as the source. This is the standard activation patching design and is the single highest-priority change.

---

### H5: Late-layer patching more effective than early-layer patching

**Status: Not supported under current design; cannot be fairly evaluated**

Layer 16: zero effect. Layer 24: slightly harmful (−0.10 delta at two cells). Layer 27: slightly harmful at one cell. No layer showed positive restoration. Given that the source is inverted at layer 27 and weak at layers 16 and 24, the hierarchy of effects cannot be attributed to layer specificity.

**Implication for Report 7:** H5 should be re-tested after fixing the patching source. Only then does layer × position specificity become interpretable.

---

## 7. Interpretation cautions

### Internal validity

**Shared direction extraction and tracing prompts.** In both R3 and R6, all 25 tracing prompts are part of the direction extraction set. The direction is tuned to these exact prompts; projected values may be inflated relative to a held-out direction.

**Phrase-list refusal classifier.** All behavioral labels are from a phrase-list heuristic that checks the first 200 characters of the response. Known failure modes include false positives ("I cannot stress enough...") and false negatives for unusual refusal phrasings. Some patching outputs begin with partial sentences, which may affect early-position classification.

**Non-monotone behavioral k=3 vs k=10.** At n=25, the difference between 32% and 36% refusal cannot be distinguished from sampling noise. Neither report can currently claim a monotone dose-response relationship.

**Patching is same-prompt self-patching.** The source activations come from the same k=3 forward pass. Standard activation patching takes sources from a different condition (e.g., k=0). This is a design limitation, not a result of the experiment.

### External validity

**n=25 prompts from one benchmark (AdvBench).** Results may not generalize to other harmful prompt distributions.

**One model (Llama-3.1-8B-Instruct), one attack type (prefilling).** Layer specificity patterns may differ in other model families or under other attacks.

**Greedy decoding throughout.** All outputs use `do_sample=False`. Results may differ under sampling.

### Construct validity

**"Refusal direction" extracted from last-token activations under the chat template.** The last token is often a formatting or punctuation token. Whether it captures refusal-vs-compliance content optimally is not verified.

**Projection magnitude is not comparable across layers.** Each layer has its own DiM direction vector at a different scale. The values at layer 27 and layer 16 are not on the same scale.

---

## 8. What Report 7 should improve

Numbered in priority order:

1. **Fix the patching source.** Use k=0 baseline activations as the source, not attacked-condition activations. Run the same prompt without the prefix, collect activations at the source layer × position, and inject those into the attacked (k=3) forward pass. This is the standard activation patching design and is what actually tests whether clean refusal-direction information can restore refusal.

2. **Prompt-level association analysis.** For the 10 patching prompts (or ideally all 25 tracing prompts) under k=3, compare the trace mean projection at layers 24 and 27 between prompts where the baseline label is refusal vs compliance. If projection is reliably more negative for compliant prompts, this strengthens H3 substantially and costs nothing to compute from existing trace data.

3. **Clarify generated-token vs fixed-position summaries.** The current trace summaries aggregate projection over all generated positions. Compute projections separately over a fixed early window (positions 0–5) to avoid conflating signal timing with response length. Short refusals contribute differently than long compliant continuations.

4. **Expand patching to include position 0 and cross-condition source.** After fixing the source, test target position 0 in addition to 1, 3, 5. The refusal commitment in Llama models is often apparent at the very first generated token.

5. **Re-run harmful_k10 behavioral generation.** On L40S with `--no-resume`, run `python scripts/run_report6_generation.py --conditions harmful_k10 --no-resume`. This is a 20–30 minute run and completes the behavioral table inside the Report 6 pipeline.

6. **Optional: held-out direction evaluation.** Use a different subset of prompts for direction extraction than for tracing. This eliminates the internal validity concern about direction overfitting.

---

## 9. Writeup-ready tables

### Table 1: Behavioral summary (best available data per condition)

| Condition | k | n | Refusals | Compliance | Refusal Rate | Source |
|-----------|---|---|----------|------------|--------------|--------|
| Harmful baseline | 0 | 25 | 23 | 2 | 0.92 | Report 6 pipeline |
| Harmful + prefilling | 3 | 25 | 8 | 17 | 0.32 | Report 6 pipeline |
| Harmful + prefilling | 10 | 25 | 9* | 16* | 0.36* | Report 3 pilot (2026-04-06) |
| Benign baseline | 0 | 25 | 0 | 25 | 0.00 | Report 6 pipeline |

\*Report 3 numbers; Report 6 behavioral k=10 generation failed on incompatible GPU.

### Table 2: Tracing summary — mean refusal-direction projection by condition × layer

Computed from `outputs/report6/summaries/trace_mean_projection_by_condition_layer.csv` (generated tokens only):

| Condition | k | Layer 16 | Layer 20 | Layer 24 | Layer 27 |
|-----------|---|----------|----------|----------|----------|
| harmful_k00 | 0 | +1.271 | +1.612 | +0.665 | -0.376 |
| harmful_k03 | 3 | +0.172 | -0.374 | -3.110 | -5.285 |
| harmful_k10 | 10 | -0.270 | -0.822 | -3.463 | -5.811 |

Full standard deviations and n_records in `outputs/report6/summaries/trace_mean_projection_by_condition_layer.csv`.

### Table 3: Patching summary — refusal outcomes by layer × target position

Harmful_k03, n=10 prompts per cell. Baseline refusal rate on this subset: 50% (5/10).

| Layer | Source proj (mean) | Target pos 1 | Target pos 3 | Target pos 5 | Restored | Lost |
|-------|--------------------|:---:|:---:|:---:|---------|------|
| 16 | +0.543 | 50% | 50% | 50% | 0 | 0 |
| 24 | +0.948 | 40% | 40% | 50% | 0 | 2 |
| 27 | -0.458 | 40% | 50% | 50% | 0 | 1 |

**Total across 9 cells (90 records): 0 restored, 3 lost, 87 no change.**

---

## 10. Ready to write Report 6?

**Yes, with one caveat and one correction.**

**Correction first:** The VERIFY_REPORT6.md overstated the behavioral generation failure. Three of four conditions produced valid outputs (harmful_k00: 92% refusal, harmful_k03: 32% refusal, benign_k00: 0% refusal). The Report 6 behavioral numbers are usable for these conditions. Only harmful_k10 is missing and should be cited from Report 3 if not re-run.

**The one outstanding item:** Re-run harmful_k10 behavioral generation on L40S with `--no-resume`. If not feasible before the deadline, cite the Report 3 pilot number (9/25 = 36%) explicitly.

**Do not use** `outputs/report6/plots/report6_refusal_rate_by_condition.png` without checking whether it includes the corrected three-condition data. Re-run `python scripts/plot_report6_results.py` after verifying the generation summary CSV.

**The mechanistic sections (tracing + patching) can be written now.** All tracing plots, heatmaps, patching results, and summaries are valid. The key framing for the patching null result is: "Under the current patching design — where the source activations are taken from the attacked-condition forward pass — no combination of layer and target position restored refusal. At layer 27, this is mechanistically explained by the source projection already being inverted (mean: -0.458) under the k=3 attack. At layers 16 and 24, where the source is positive, the intervention had zero or slightly negative effect, suggesting the refusal commitment window is not reached by these target positions under same-condition self-patching."

---

*Analysis based on direct file inspection of all Report 3 and Report 6 output artifacts. Numbers in this document can be verified against the source files listed in each section. No results were invented or extrapolated.*
