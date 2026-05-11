# Critique of `claude_code_prompt_report7_top_tier.md`

A read of the Report 7 prompt against (a) Report 6's actual gaps, (b) `report7_plan.md`, and (c) the time/complexity budget for one course report.

---

## TL;DR — recommended Report 7 plan

The prompt is **mostly the right shape** and adds one thing the plan was missing (the additive-direction intervention, which is a real upgrade). But it also asks for too many artifacts, ignores two validity threats the plan correctly flags, and reintroduces the doc-file zoo we just archived.

**Do this much for Report 7:**

1. Hold out the refusal direction (re-extract on a disjoint prompt set).
2. Re-run k=10 generation on a compatible GPU.
3. Prompt-level H3 association test (no compute).
4. Cross-condition patching with k=0 source → k=3 target, focused on layers {24, 27} and target positions {0, 1, 3, 5}, n=25 prompts, **one** primary source position (last input token).
5. Additive refusal-direction intervention at layers {24, 27}, alphas {1.0, 2.0}, target positions {0, 1, 3}, n=25, with **one** control (layer 16 baseline; add random-direction only if cheap).
6. Bootstrap CIs on every cell-level rate; McNemar where discordant pairs ≥ 5.
7. One comparison table (R6 same-condition vs. R7 cross-condition vs. R7 additive).
8. Write the discussion using the prompt's "scientific interpretation rules" verbatim — those rules are excellent.

**Skip or defer the rest.** Detailed reasoning below.

---

## What the prompt gets right (do these)

### 1. The cross-condition patching fix (H4)
Prompt's `harmful_k00 → harmful_k03` cross-condition design is exactly the fix Report 6 flagged in Section 7.2. Same as `report7_plan.md` Change 1. **Highest-priority experiment of Report 7.**

### 2. The additive-direction intervention (H5)
This is the **single best addition the prompt makes over `report7_plan.md`**. Adding `α · refusal_direction[layer]` to the attacked residual is a cleaner sufficiency test than activation patching:
- Patching tests "can a clean state restore refusal?" — confounded by everything else in that state.
- Additive tests "is *just the refusal-direction component* sufficient?" — isolated.

Together, the two interventions form a real causal staircase. Keep this.

### 3. Prompt-level H3 association
Free win. Both documents agree. Run it first because it's cheap and may inform which prompts to use for patching.

### 4. Controls (H6) and the "no overclaim" interpretation rules
The interpretation rules section (lines 540–574 of the prompt) is the strongest writing in the document. Those exact rules should land in the report's Discussion section. The control list (random/orthogonal/shuffled/layer-16) is well-framed.

### 5. Scope-control section
The "do NOT implement unless explicitly requested" list (suffix attacks, attention heads, neurons, all 32 layers, multiple models, training) is correct and matches the project's scope.

---

## What the prompt gets wrong or omits

### A. Missing: hold out the refusal direction
`report7_plan.md` Change 3 calls this out explicitly. Report 6 extracted the direction from 50 prompts and traced 25 of those *same* prompts. That's a methodological hole big enough to weaken H2 even though H2 is otherwise the project's strongest result.

The prompt does not mention this. **Add it.** Re-extracting on a disjoint set is ~10 minutes of compute and the easiest way to make the H2 result robust.

### B. Missing: k=10 generation rerun
Report 6 had to borrow the k=10 row from Report 3's pilot due to a GPU compatibility error. The plan's Change 4 fixes this; the prompt makes k=10 "optional." It should be **required** — a complete, single-pipeline behavioral table is what makes the report look professional.

### C. The doc-file zoo it asks to recreate
Lines 178–182 ask for `RUN_REPORT7.md`, `VERIFY_REPORT7.md`, `BUILD_JOURNAL_REPORT7.md`, `REPORT7_EXPERIMENT_PLAN.md` — the exact four files we just archived because they accumulated cruft. **Don't recreate them.** Use:
- One consolidated `archive/REPORT7_NOTES.md` (or just keep notes in conversation/PR descriptions).
- The verification logic should live in a script that prints a status summary, not a tracked markdown.

### D. Statistical inference is punted twice
Both documents say "no fake p-values." Fine, but the prompt then proposes a 250–500 record patching grid with no plan for quantifying noise. **Bootstrap CIs and McNemar's test on paired (per-prompt, per-cell) outcomes are cheap and should be in the must-do list**, not optional. Without them, a "20% restoration rate" with n=10 looks identical to a "20% restoration rate" with n=25 in the report.

### E. The additive intervention grid is too large at default
Default ask: 3 layers × 3 alphas × 4 positions × 10 prompts × 4 direction types = **1,440 generations**. Even at 25 prompts that's 3,600. Cut to:
- Layers {24, 27} (drop 16 from intervention runs; use it as a control only)
- Alphas {1.0, 2.0} (drop 0.5; if it does nothing, no information)
- Positions {0, 1, 3} (drop 5 unless 0/1/3 show signal)
- One control direction (random) + layer-16 comparison

That's 2 × 2 × 3 × 25 = 300 generations + a comparable control budget. Tractable.

### F. The "secondary source position" experiment from the plan
`report7_plan.md` Change 6 proposes a second source variant (k=0 source at *generated position 0*). This isolates whether the restoration signal lives in the pre-generation state or the early-generation commitment. It's interesting but **defer it**. The additive intervention covers similar conceptual ground more cleanly.

### G. The discriminator framing is implicit
`report7_plan.md` makes the information-loss (H6) vs. active-suppression (H7) discrimination explicit. The prompt's H4 + H5 implicitly do this — if H5 (additive) succeeds, that's information-loss; if it fails, that's active-suppression. **State this explicitly in the Hypotheses section** so the report has a clean headline interpretation regardless of outcome.

---

## Why not just do everything the prompt asks for

| Cost | Concrete impact |
| --- | --- |
| Compute | Prompt's full default grid is ≥3,600 generations across patching + additive + controls. On A100 at ~1s/generation that's ~1 hour just for inference, plus tracing. With k=10 + held-out direction added, it grows. |
| Output sprawl | 8 plots, 4 doc files, ~5 new scripts, 4 new configs, ~10 CSV summaries. The prompt acknowledges this risk but still asks for it. Most plots will not appear in the 6-page report. |
| Re-stale-ification | Recreating BUILD_JOURNAL/RUN/VERIFY/PLAN markdowns rebuilds the exact mess we just cleaned. |
| Validity threats untouched | Without the held-out direction (A) and the k=10 rerun (B), no amount of new intervention work fixes Report 6's known weaknesses. |
| Statistical legibility | A grid this size with no CIs/tests buries the actual signal in noise. The reader cannot tell which cells are real. |

The version above (TL;DR) achieves all of the prompt's substantive goals with about 40% of the artifacts and adds the two validity fixes the prompt missed.

---

## What to keep verbatim from the prompt

- The Revised Research Question (lines 65–71).
- H1–H6 hypothesis statements (lines 75–101) — minor edit: explicitly note that H5 outcome discriminates information-loss vs. active-suppression.
- Scope-control "do NOT implement" list (lines 135–143).
- Scientific interpretation rules (lines 540–574).
- Fallback plan structure (lines 513–520).

These five blocks are the strongest content in the prompt and should drive the actual Report 7 writing.

---

## Order of operations

1. Hold-out direction re-extraction + re-trace (validates H2 first; ~30 min).
2. Prompt-level H3 (5 min, pure pandas).
3. k=10 rerun (5 min on A100).
4. Cross-condition patching (primary experiment; ~1–2 hr).
5. Additive intervention (focused grid; ~1–2 hr).
6. Comparison table + plots + write-up.

If only one experiment can run: **#4 with the primary source** is the experiment Report 7 exists to do.
