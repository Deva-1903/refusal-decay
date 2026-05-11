# Report 7 — Plan

Report 6 ended with a specific punch list (Section 7.2 "Priorities for Report 7") and a clear unresolved question (information-loss vs. active-suppression). This plan turns that into concrete experiments, analyses, and writing changes.

---

## 1. What stays, what changes from Report 6

### Stays
- System: Llama-3.1-8B-Instruct, 32 layers, hidden 4096.
- Task: safety-constrained generation; phrase-classifier refusal label on first ~200 chars.
- Environment: prefilling attacks at k ∈ {0, 3, 10} on AdvBench harmful prompts + Alpaca benign controls.
- Tracing layers: {16, 20, 24, 27}.
- Refusal-direction extraction: difference-in-means at each layer.
- H1 (behavioral attack effect) and H2 (late-layer specificity) — already supported; expand sample.

### Changes (the substantive work of Report 7)

**Change 1 — Fix the patching source.** This is the #1 priority Report 6 already identified.
- Old: source = last input/prefill token from the *same k=3 attacked* forward pass. At layer 27 this source had projection −0.458, so it was an attacked-into-attacked patch.
- New: source = last input/prefill token from the *k=0 unattacked* forward pass of the **same prompt**. This is the standard activation-patching design.
- This single change converts H4/H5 from "not fairly evaluable" to "actually testable."

**Change 2 — Add the prompt-level H3 test (no new compute).**
- Within the 25 harmful k=3 prompts, split into refusers (8) vs. compliers (17).
- Test: do refusers have less-negative mean layer-24 and layer-27 projection than compliers? Paired across prompts.
- Free win — data already exists in `traces_all.parquet`.

**Change 3 — Hold out the refusal direction.**
- Old: 25 traced prompts were a subset of the 50 used for direction extraction.
- New: re-extract refusal direction from a disjoint set (50 harmful + 50 benign prompts that do NOT appear in the traced subset). Re-run tracing. Verify the late-layer monotone gradient still holds — if it does, the H2 result is robust.

**Change 4 — Re-run k=10 generation on a compatible GPU.**
- Required so the entire behavioral table is from one pipeline. Closes the gap that forced Report 6 to borrow the k=10 row from Report 3.

**Change 5 — Expand the patching grid where it matters.**
- Drop layer 16 from the patching grid (Report 6 already shows the effect isn't there).
- Concentrate on layers {24, 27}.
- Expand target positions to {0, 1, 3, 5, 8} — Report 6 noted that refusal commitment often happens at the very first generated token, but position 0 wasn't tested.
- Increase patching prompts from 10 → 25 (use the same set as tracing, so prompt-level association can be tied back to patching outcome per-prompt).
- Grid: 2 layers × 5 target positions × 25 prompts = 250 records (≈3× R6's 90).

**Change 6 — Add a second source-position variant.**
- Primary: k=0 source at last input token (Change 1).
- Secondary: k=0 source at *generated position 0* of the unattacked forward pass.
- This isolates whether what matters is the "pre-generation" state or the early-generation refusal commitment.

**Change 7 — Add the information-loss vs. active-suppression discriminator.**
- This is what Report 6 left unresolved.
- The discriminator: with a *clean k=0 source*, does patching restore refusal?
  - **Yes →** the attacked state was missing the signal (information-loss supported).
  - **No →** even injecting a clean refusal signal is overridden downstream (active-suppression supported).
- This is exactly the question H4 was trying to ask; Report 6 just had the wrong source.

---

## 2. Revised hypotheses

| H | Statement | Status entering R7 |
| --- | --- | --- |
| H1 | k=3 prefilling reduces refusal rate vs. k=0 on harmful prompts. | Supported in R6 (0.92 → 0.32). Confirm on full set + add k=10. |
| H2 | The refusal-direction projection becomes more negative under attack in late layers (24, 27) than in comparison layers (16, 20). | Supported in R6 (monotone Δ gradient 1.10 → 1.99 → 3.78 → 4.91). Re-verify with held-out direction. |
| H3 | Within k=3, prompts that refuse have less-negative late-layer projection than prompts that comply. | Untested at prompt level in R6. Run the test. |
| **H4-revised** | **Patching the refusal-direction component from a k=0 source into attacked late-layer positions increases refusal rate vs. unpatched attacked baseline.** | Not fairly tested in R6 (attacked source). Re-test with clean source. |
| **H5-revised** | **Conditional on H4: restoration effects are larger at layers 24/27 than at earlier layers** (and at certain target positions, especially position 0). | Not fairly tested in R6. Re-test. |
| **H6-new (information-loss)** | If H4 is supported with a clean source, the attack works primarily by *removing* the late-layer refusal signal. | New for R7. |
| **H7-new (active-suppression)** | If H4 fails even with a clean source, downstream computations *override* the injected refusal signal — the attack works by adding a suppressive contribution, not by deleting the safety contribution. | New for R7. |

H6 and H7 are mutually exclusive given the same experiment. That is the point: the design distinguishes them.

---

## 3. Experiments to run

1. **Re-extract refusal direction on held-out 50+50 prompts.** Save the new `refusal_direction_heldout.pt`. ~10 min on A100.
2. **Re-run tracing** with held-out direction across k ∈ {0, 3, 10}, 25 traced harmful prompts × 4 layers × generated positions. ~30 min.
3. **Re-run k=10 generation** on a compatible GPU (NOT the GTX 1080 Ti node from R6). ~5 min.
4. **Prompt-level H3 test.** Pure pandas/scipy — no model calls. ~5 min.
5. **Patching grid (clean source).** 2 layers × 5 target positions × 25 prompts × 2 source variants = 500 records. Largest compute item. Estimate ~2 hours on A100. If tight, drop the secondary source variant first; do not drop position 0.
6. **Patching effect-size analysis.** Per-cell refusal-rate Δ vs. attacked baseline, with bootstrap CIs. McNemar's test if cell n permits.

---

## 4. Analyses

- **Behavioral table** (Table 1): complete the k=10 row from the same pipeline.
- **Tracing table + heatmap** (Table 2, Figure 2): re-do with held-out direction. Confirm monotone gradient.
- **Prompt-level scatter:** layer-27 mean projection per prompt vs. behavioral refusal under k=3. Color refusers vs. compliers. (Figure 5, new.)
- **Patching grid heatmap** (Figure 6, new): refusal rate by (target layer, target position) under clean-source patching. Side-by-side with R6's attacked-source result for direct contrast.
- **Per-prompt patch effect plot:** for each of the 25 prompts, did clean-source patching flip behavior at the best (layer, position) cell? Tied back to per-prompt H3 projection.
- **Effect-size summaries:** refusal-rate Δ + bootstrap CIs for every cell. McNemar's only where discordant pairs ≥ 5.

---

## 5. Updated threats to validity to write up

- **Phrase-classifier limits** (kept from R6). Add: spot-check 20 borderline outputs manually; report any flips.
- **Direction overfitting** (addressed by Change 3).
- **Patching source contamination** (addressed by Change 1).
- **Small patching n** (partially addressed by Change 5: 10 → 25).
- **Single model, single attack family** (acknowledged; not fixable in R7 scope).
- **Layer-specific projection magnitudes not comparable across layers** (kept from R6).

---

## 6. Skeleton of the revised paper

1. Revised framing (1 page).
2. High-level design — 2 blocks (tracing + clean-source patching), unchanged structure but cleaner motivation.
3. Behavioral results (Table 1, complete this time).
4. Tracing results (Table 2, held-out direction; Figure 2 heatmap; Figure 3 projection-vs-position).
5. **NEW: Prompt-level association (H3) — Figure 5.**
6. **REVISED: Patching results with clean source — Table 3, Figure 4, Figure 6.**
7. **NEW: Discrimination between information-loss and active-suppression — the headline result of R7.**
8. Updated hypothesis status table (Table 4).
9. Conclusions, threats to validity.

---

## 7. Order of operations (what to run first)

1. **H3 prompt-level test** — costs ~5 min, settles a Report 6 caveat immediately, may also reveal which prompts are good patching candidates.
2. **Held-out direction re-extraction + re-trace** — confirms or weakens H2, our strongest result.
3. **k=10 re-run** — closes a known gap, easy.
4. **Clean-source patching grid** — the main experiment. Run with primary source (last-input k=0) first; add secondary source (gen-pos-0 k=0) only if time permits.

If only one of these can run, it is **#4 with the primary source** — that is the experiment Report 7 actually exists to perform.
