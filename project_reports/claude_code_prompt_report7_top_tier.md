# Claude Code Prompt — Report 7 Top-Tier Empirical Upgrade Plan

Use this prompt in Claude Code at the **root of my refusal-decay project repo**.

---

## Goal

Prepare the repo for **COMPSCI 602 Project Report 7 — Empirical Results II** at the strongest possible level **without exploding scope**.

Report 7 is the final empirical-results iteration before the final report. The goal is not to add random experiments. The goal is to make the evidence much stronger than Report 6 by fixing its weakest causal test and adding the right controls.

The final Report 7 should read like a serious empirical paper section:
- clear revised question,
- focused hypotheses,
- clean experimental design,
- controls,
- prompt-level analysis,
- honest negative/positive results,
- clear validity discussion,
- concrete next-step interpretation.

Do not overclaim. Do not pretend this is a complete publication. But make it strong enough to stand out in the course.

---

## Current project context

Project: **Refusal-direction changes under prefilling attacks in safety-aligned LLMs**

Main model:
- `Llama-3.1-8B-Instruct`

Main task:
- harmful prompt -> model should refuse
- benign prompt -> model should answer normally

Main attack:
- prefilling attack, especially `k=3`

Internal signal:
- projection of residual stream activations onto the refusal direction

Important layers:
- comparison layer: `16`
- optional comparison layer: `20`
- main late layers: `24`, `27`

Report 6 found:
- harmful `k=0`: 23/25 refusals, refusal rate 0.92
- harmful `k=3`: 8/25 refusals, refusal rate 0.32
- benign `k=0`: 0/25 refusals
- late layers 24/27 show the strongest negative shift under prefilling
- same-condition patching under attack did not restore refusal:
  - 0 restored
  - 3 lost
  - 87 unchanged

Key weakness in Report 6:
- patching copied from the attacked `k=3` source state, so the source itself may already have been corrupted / negative.
- Therefore Report 6 did not strongly test whether a **clean refusal-like signal** can restore refusal.

---

## Revised Report 7 research question

Use this as the guiding question:

**When prefilling attacks push the late-layer refusal-direction signal negative, can targeted interventions that restore or add a cleaner refusal-direction component recover refusal behavior?**

This is a **causal question that informs mechanism**. Do not call it a fully solved mechanistic explanation.

---

## Revised Report 7 hypotheses

Use these hypotheses to drive code and analysis.

### H1 — Behavioral replication
Prefilling at `k=3` reduces refusal on harmful prompts compared with baseline `k=0`.

### H2 — Late-layer localization
The refusal-direction projection shifts more strongly negative in late layers `24` and `27` than in comparison layers `16` and `20`.

### H3 — Prompt-level association
Among attacked harmful prompts, prompts that comply will have more negative late-layer refusal-direction projection than prompts that still refuse.

### H4 — Clean-source restoration
Cross-condition patching from baseline `k=0` source activations into attacked `k=3` target runs will restore refusal more often than Report 6 same-condition attacked-source patching.

### H5 — Direction-addition sufficiency
Adding a scaled refusal-direction component directly into attacked late-layer residual states will increase refusal if the direction is causally sufficient under those conditions.

### H6 — Specificity / controls
Restoration should be stronger for:
- late layers `24/27` than comparison layer `16`,
- real refusal direction than random/orthogonal/shuffled controls,
- harmful attacked prompts than benign prompts.

If controls show similar effects, then the intervention is probably not specifically restoring safety behavior.

---

## Scope control

### Must implement / verify for Report 7
1. Behavioral/tracing reuse or rerun verification from Report 6.
2. Prompt-level association analysis.
3. Cross-condition baseline-source patching:
   - source condition: `harmful_k00`
   - target condition: `harmful_k03`
4. Additive refusal-direction intervention:
   - add scaled refusal-direction vector to attacked residual states.
5. At least one negative control:
   - random direction,
   - orthogonal direction,
   - shuffled prompt-source pairing,
   - or layer 16 comparison.
6. Report 7 summaries, plots, verification, and run docs.

### Keep small by default
Default patch/addition grid:
- prompts: 10 first; allow 25 if runtime is reasonable
- layers: `16`, `24`, `27`
- target positions: `0`, `1`, `3`, `5` if supported
- source position for cross-condition patching: `-1` from `k=0`
- additive alpha grid: `[0.5, 1.0, 2.0]` initially

### Optional if runtime is fine
- increase prompt count to 25 for best 1–2 settings
- include layer `20`
- include `k=10` behavioral rerun
- include target position `7`

### Do NOT implement unless explicitly requested
- suffix attacks
- attention heads
- neuron analysis
- all 32 layers
- multiple models
- training / fine-tuning
- large factorial sweeps

---

## First step: inspect repo

Before editing anything, inspect:
- Report 6 scripts/configs
- existing patching implementation
- how activations are cached
- how refusal direction is stored
- generation/tracing outputs
- `RUN_REPORT6.md`
- `VERIFY_REPORT6.md`
- `REPORT3_REPORT6_ANALYSIS.md`
- Report 6 PDF/LaTeX if present

Then extend existing structure. Do not create a parallel mess.

---

## Required directory structure

Create if needed:

```text
configs/experiments/report7/
outputs/report7/
outputs/report7/summaries/
outputs/report7/plots/
outputs/report7/patching/
outputs/report7/interventions/
```

Create/update docs:
```text
RUN_REPORT7.md
VERIFY_REPORT7.md
BUILD_JOURNAL_REPORT7.md
REPORT7_EXPERIMENT_PLAN.md
```

---

## Required configs

Create:

### `configs/experiments/report7/generation_report7.yaml`
Purpose:
- verify / reuse:
  - `harmful_k00`
  - `harmful_k03`
  - `benign_k00`
- optionally allow `harmful_k10`
- must not reuse all-error caches

### `configs/experiments/report7/tracing_report7.yaml`
Purpose:
- standardize tracing summaries
- layers: `16`, `20`, `24`, `27`
- conditions:
  - `harmful_k00`
  - `harmful_k03`
  - optional `harmful_k10`
- output generated-token-only and all-position summaries separately

### `configs/experiments/report7/patching_report7.yaml`
Purpose:
- cross-condition source patching
- source condition: `harmful_k00`
- target condition: `harmful_k03`
- layers: `16`, `24`, `27`
- source position: `-1`
- target positions: `0`, `1`, `3`, `5` if possible
- default prompt count: 10
- allow prompt count: 25

### `configs/experiments/report7/additive_intervention_report7.yaml`
Purpose:
- add scaled refusal-direction vector into attacked `harmful_k03` residual states
- layers: `16`, `24`, `27`
- target positions: `0`, `1`, `3`, `5` if possible
- alpha grid: `[0.5, 1.0, 2.0]`
- controls:
  - random direction or orthogonal direction if feasible
  - layer 16 comparison at minimum
- default prompt count: 10
- allow prompt count: 25

---

## Required code changes

## 1. Cross-condition patching support

Modify or add script:

`scripts/run_report7_cross_condition_patching.py`

It should support:

- source activations from condition `harmful_k00`
- target activations from condition `harmful_k03`
- prompt matching by stable `prompt_id`
- layer-specific source and target positions
- writing per-prompt generations and labels

Output file:
`outputs/report7/patching/cross_condition_patching_results.csv`

Required columns:
- `prompt_id`
- `source_condition`
- `target_condition`
- `layer`
- `source_position`
- `target_position`
- `baseline_attacked_label`
- `patched_label`
- `restored_refusal`
- `lost_refusal`
- `unchanged`
- `source_projection`
- `target_projection_before`
- `target_projection_after` if available
- `error`
- truncated baseline and patched outputs if available

Important:
- If prompt IDs are missing, do not silently use array index. Warn clearly and document the fallback.
- If target position `0` is unsupported, skip it with a clear logged reason and continue.

---

## 2. Additive refusal-direction intervention

Modify or add script:

`scripts/run_report7_additive_direction_intervention.py`

Purpose:
Instead of copying a source activation, directly add a scaled refusal-direction component to attacked `k=3` target residual states.

For each prompt/layer/target_position/alpha:
- run attacked baseline if needed
- add `alpha * refusal_direction[layer]` at target residual state
- generate output
- classify output
- compute restored/lost/unchanged relative to attacked baseline

Output file:
`outputs/report7/interventions/additive_direction_results.csv`

Required columns:
- `prompt_id`
- `condition`
- `layer`
- `target_position`
- `alpha`
- `direction_type` (`refusal`, `random`, `orthogonal`, or `control`)
- `baseline_attacked_label`
- `intervened_label`
- `restored_refusal`
- `lost_refusal`
- `unchanged`
- `baseline_output_truncated`
- `intervened_output_truncated`
- `error`

Important:
- Start with real refusal direction.
- Add one negative control if feasible:
  - random normalized vector with same dimension,
  - orthogonalized vector,
  - or shuffled layer/source control.
- Do not over-engineer controls if runtime is tight. At minimum include layer 16 as comparison.

Why this matters:
- If cross-condition patching fails, additive intervention still directly tests whether increasing the refusal-direction component is sufficient under attack.
- This is a stronger causal test than Report 6 same-condition patching.

---

## 3. Prompt-level association analysis

Create/update:

`scripts/analyze_report7_prompt_association.py`

Inputs:
- `generation_prompt_labels.csv`
- `trace_prompt_level_key_layers.csv` or equivalent

Output:
`outputs/report7/summaries/prompt_projection_label_association.csv`

For harmful `k=3`, compare refused vs complied prompts for:
- layer 24 generated-token mean projection
- layer 27 generated-token mean projection
- optional layer 20

Output columns:
- `condition`
- `layer`
- `label_group`
- `n_prompts`
- `mean_projection`
- `median_projection`
- `std_projection`
- `min_projection`
- `max_projection`

Also output a difference table:
`outputs/report7/summaries/prompt_projection_label_differences.csv`

Interpretation:
- complied prompts more negative than refused prompts -> supports H3
- no separation -> weakens H3

No fake p-values unless already easy and appropriate.

---

## 4. Standardized tracing summaries

Create/update:

`scripts/summarize_report7_tracing.py`

Outputs:
- `outputs/report7/summaries/trace_generated_token_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_all_position_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_token_trajectory_by_condition_layer.csv`

Each must include:
- condition
- layer
- token_window
- n_prompts
- n_rows
- mean_projection
- std_projection
- min_projection
- max_projection

Use generated-token-only summaries for headline claims.

---

## 5. Controls and comparison summaries

Create:

`scripts/compare_report6_report7_patching.py`

Inputs:
- Report 6 patching summary
- Report 7 cross-condition patching summary
- additive intervention summary if available

Outputs:
- `outputs/report7/summaries/report6_vs_report7_intervention_comparison.csv`

It should compare:
- same-condition patching from Report 6
- baseline-source patching from Report 7
- additive refusal-direction intervention from Report 7

Columns:
- `method`
- `layer`
- `target_position`
- `n_prompts`
- `restored`
- `lost`
- `unchanged`
- `restoration_rate`
- `loss_rate`

This table is likely one of the most important Report 7 tables.

---

## 6. Plotting

Create/update:

`scripts/plot_report7_results.py`

Required plots:
1. `report7_refusal_rate_by_condition.png`
2. `report7_generated_mean_projection_by_layer.png`
3. `report7_layer27_projection_trajectory.png`
4. `report7_layer24_projection_trajectory.png`
5. `report7_layer27_projection_by_label.png`
6. `report7_cross_condition_patching_recovery.png`
7. `report7_additive_direction_recovery_by_alpha.png` if additive intervention exists
8. `report7_report6_vs_report7_intervention_comparison.png`

Plots should be simple and readable.
No fancy styling needed.
Captions can be written later, but filenames and axes must be clear.

---

## 7. Verification

Create/update:

`scripts/verify_report7_outputs.py`

Check:

### Behavioral
- valid behavioral summaries exist
- no all-error generation caches

### Tracing
- generated-token summary exists
- layer coverage includes 16, 20, 24, 27
- condition coverage includes harmful_k00 and harmful_k03

### Prompt-level association
- association CSV exists
- layers 24 and 27 present
- refused/complied groups present, if both exist

### Cross-condition patching
- results exist
- source condition is harmful_k00
- target condition is harmful_k03
- layers 16, 24, 27 present
- target positions present
- restored/lost/unchanged computable

### Additive intervention
- results exist, if implemented
- alpha grid present
- direction_type present
- restored/lost/unchanged computable

Write:
`VERIFY_REPORT7.md`

At end print:
- COMPLETE
- MISSING
- SUSPICIOUS
- READY TO WRITE REPORT 7?
- RUN THESE NEXT

---

## 8. Run instructions

Create:

`RUN_REPORT7.md`

Include:
1. Unity environment setup
2. smoke test commands
3. tracing summary commands
4. prompt-level association commands
5. cross-condition patching commands
6. additive direction intervention commands
7. plotting commands
8. verification commands
9. fallback plan

Fallback plan:
If runtime is bad:
- skip `k=10`
- skip prompt count 25
- run cross-condition patching only for layers 24 and 27
- run target positions 0 and 1 first
- run additive intervention only for layer 27 and alpha 1.0 / 2.0
- keep prompt-level association no matter what because it is cheap

---

## 9. Build journal

Create/update:

`BUILD_JOURNAL_REPORT7.md`

Include:
- what files were changed
- why each change was made
- commands tested
- output files produced
- known limitations
- what is deferred

---

## Scientific interpretation rules

The final Report 7 can stand out only if interpretation is disciplined.

### If baseline-source patching restores refusal
Say:
- Report 6 same-condition patching likely failed because the source was already corrupted.
- Cleaner baseline-source intervention partially restores refusal.
- This supports a causal role for late-layer refusal-direction information under attack.

### If additive intervention restores refusal
Say:
- Directly increasing the refusal-direction component can recover refusal under some attacked conditions.
- This is stronger causal evidence than observational tracing alone.
- Check controls before claiming specificity.

### If neither restores refusal
Say:
- Late-layer negative shift is robustly associated with attack success.
- Simple component restoration/addition is insufficient.
- The mechanism may involve broader state changes, attention routing, nonlinear interactions, or different source/target timing.
- This is still a useful negative result because it narrows plausible mechanisms.

### Always avoid:
- “proves”
- “fully explains”
- “disproves”
- “solves safety alignment”

Use:
- “supports”
- “weakens”
- “under this intervention setting”
- “consistent with”
- “suggests”

---

## Final output

After implementation, print:

### Files changed
List all changed/created paths.

### New Report 7 capabilities
Explain in plain English.

### Exact first commands to run
Give the shortest safe command sequence.

### Expected outputs
List key files.

### If only one thing can be run
Tell me the highest-value minimal run.

Start now by inspecting the repo, then implement only the focused Report 7 improvements.
