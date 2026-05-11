# Claude Code Prompt — Analyze Report 3 and Report 6 Results Into One Research Summary Document

Use this prompt in Claude Code at the **root of my project repo**.

---

You are helping me analyze and summarize the progression from **COMPSCI 602 Project Report 3** to **Project Report 6** for my project:

**Tracing refusal-direction changes under prefilling attacks in safety-aligned LLMs**

The goal is to create a single, detailed, writeup-ready markdown document that explains:

1. what we did in Report 3,
2. what we planned after Report 3,
3. what we did in Report 6,
4. what changed between Report 3 and Report 6,
5. what the results currently show,
6. what is genuinely new in Report 6,
7. what remains unresolved,
8. what should be improved in Report 7.

This document is for me to understand the experimental story before writing Report 6 and Report 7. Be skeptical, concrete, and evidence-based. Do not invent results.

---

## Project context

This project studies how **prefilling attacks** affect the model’s refusal behavior and the internal **refusal-direction signal**.

The main model is:

- `Llama-3.1-8B-Instruct`

The main task is:

- harmful prompt → model should refuse
- prefilling attack → model is forced to begin with a compliant prefix

The main mechanism being tracked is:

- projection of residual stream activations onto a refusal direction

Important layers:

- comparison layers: `16`, `20`
- main late layers: `24`, `27`

Important conditions:

- harmful baseline: `k=0`
- harmful prefilling attack: `k=3`
- optional attack: `k=10`
- benign control: `benign k=0`

---

## High-level known story so far

### Report 3
Report 3 was an exploratory phenomena report.

It found:

- harmful baseline refusal was high
- prefilling sharply reduced refusal
- the attack effect appeared by `k=3`
- longer prefilling (`k=10`) did not obviously increase behavioral attack success much beyond `k=3`
- internally, the refusal-direction projection shifted strongly negative in late layers, especially layers around `24` and `27`
- a tiny patching pilot at layer `27` did not restore refusal

Report 3’s purpose was not to finish the whole project. It was to discover the important pattern and narrow the next design.

### Report 6
Report 6 is the first focused, design-driven results report after narrowing.

Current Report 6 results appear to include:

Behavioral results:
- harmful `k=0`: 25 valid prompts, refusal rate about `0.92`
- harmful `k=3`: 25 valid prompts, refusal rate about `0.32`
- benign `k=0`: 25 valid prompts, refusal rate about `0.00`

Tracing results:
- tracing summaries exist for key layers
- late layers `24` and `27` show the strongest negative shift under prefilling
- comparison layers `16` and `20` are less affected

Patching results:
- patching was run under attacked condition `k=3`
- target layers include `16`, `24`, `27`
- target positions include `1`, `3`, `5`
- source position is likely `-1`
- patching does not restore refusal in the current setup
- some late-layer patching settings may slightly reduce refusal rather than recover it

Important caution:
- The current patching setup may copy a source state that is already attack-corrupted / negative, so the null result should not be overinterpreted as disproving the broader restoration hypothesis.

---

## What I want you to do

Please inspect the repo and produce a detailed markdown document.

Create:

`REPORT3_REPORT6_ANALYSIS.md`

The document should be written in a clear research-handoff style.

It should not just dump numbers. It should explain **what**, **why**, and **what we learned**.

---

## Step 1 — Inspect the repo

First, inspect the project structure and identify relevant files.

Look for:

### Report 3 artifacts
- Report 3 draft / PDF / LaTeX if present
- Report 3 generation outputs
- Report 3 classified outputs
- Report 3 tracing outputs
- Report 3 patching pilot outputs
- Report 3 figures/tables
- configs used for Report 3

Possible output names may include things like:
- `outputs/generations/`
- `outputs/traces/`
- `outputs/patching/`
- `sweep_classified.jsonl`
- `traces_all.parquet`
- `mean_projection_by_k_layer.csv`
- `patch_layer27...jsonl`
- `refusal_rate_vs_k.png`
- `projection_heatmap_comparison.png`

### Report 6 artifacts
- `outputs/report6/`
- `generation_refusal_rates_by_condition.csv`
- `generation_prompt_labels.csv`
- `trace_mean_projection_by_condition_layer.csv`
- `trace_projection_by_condition_layer_token.csv`
- `trace_prompt_level_key_layers.csv`
- `patching_prompt_results.csv`
- `patching_refusal_recovery_summary.csv`
- Report 6 plots
- `report6_manifest.json`
- `RUN_REPORT6.md`
- `VERIFY_REPORT6.md`
- configs under `configs/experiments/report6/`

Do not assume exact paths. Search the repo carefully.

---

## Step 2 — Build an experiment inventory

Create a table in the document called:

`Experiment inventory`

It should include columns:

- report
- experiment block
- condition(s)
- prompt count
- layers
- positions
- output files
- status
- notes

Include rows for:

### Report 3
- behavioral baseline / prefilling sweep
- refusal-direction extraction
- tracing
- patching pilot

### Report 6
- behavioral generation
- tracing
- prompt-level tracing summaries
- patching
- plotting / summaries

Clearly mark whether files are:
- found
- missing
- unclear
- derived from another run

---

## Step 3 — Summarize Report 3

Create a section:

`## Report 3: exploratory phenomena pass`

Include:

### What we were trying to do
Explain that Report 3 was meant to test experimental infrastructure and discover interesting behavior, not finish the whole project.

### What we varied
- prefilling length `k`
- likely `k=0`, `k=3`, `k=10`
- layers for tracing
- harmful prompts
- benign controls if present
- one small patching pilot

### What we measured
- refusal rate
- refusal/compliance label
- refusal-direction projection by layer and token position
- patching recovery if applicable

### What we found
Use actual repo outputs if available. If not available, use the report text if present.

Expected key findings:
- high baseline refusal
- sharp drop under prefilling
- plateau after `k=3`
- strong late-layer negative shift
- patching pilot did not restore refusal

### What was surprising
Explain:
- expected monotonic decay with longer prefilling
- instead saw behavioral plateau
- expected weakening of refusal direction
- instead saw strong negative shift / possible sign reversal in late layers

### What Report 3 changed
Explain that Report 3 narrowed the project toward:
- prefilling rather than suffix attacks
- refusal direction rather than heads/neurons
- late layers `24` and `27`
- targeted patching rather than huge sweeps

---

## Step 4 — Summarize Report 6

Create a section:

`## Report 6: focused design-driven first pass`

Include:

### What we were trying to do
Explain that Report 6 executes the narrowed design from Report 5/after Report 3.

The main question should be framed as:

**How and under what intervention conditions does the late-layer negative shift in the refusal-direction signal contribute to refusal failure under prefilling attacks?**

Call this a **causal question that informs mechanism**, not a fully mechanistic proof.

### What we ran
Summarize:

Behavioral:
- harmful `k=0`
- harmful `k=3`
- benign `k=0`
- optional `k=10` if present

Tracing:
- key layers `16`, `20`, `24`, `27`
- token-position summaries
- prompt-level summaries for key layers

Patching:
- attacked condition `k=3`
- layers `16`, `24`, `27`
- target positions `1`, `3`, `5`
- source position `-1`
- prompt subset size

### What we found
Use actual CSVs to fill in numbers.

Behavioral:
- refusal counts and rates by condition
- e.g., harmful `k=0`: 0.92, harmful `k=3`: 0.32, benign `k=0`: 0.00 if confirmed

Tracing:
- mean projection by condition and layer
- identify strongest late-layer negative shift
- compare layers `16/20` vs `24/27`
- distinguish all-position summaries vs generated-token-only summaries if both exist

Patching:
- summarize no recovery
- report any negative deltas
- identify whether late-layer patching differs from layer `16`

### What is newly learned in Report 6
This is important.

Explain clearly:
- Report 3 found the pattern
- Report 6 verifies it in a cleaner focused design
- Report 6 adds a broader structured patching pass
- Report 6 shows current patching source/target setup is not sufficient
- Report 6 helps refine Report 7 by showing that better source selection may be needed

---

## Step 5 — Compare Report 3 vs Report 6 directly

Create a section:

`## Direct comparison: Report 3 vs Report 6`

Include a table:

| Topic | Report 3 | Report 6 | What changed / what we learned |
|---|---|---|---|

Rows:

- goal
- design style
- behavioral conditions
- refusal rate pattern
- tracing layers
- late-layer pattern
- patching setup
- patching result
- interpretation
- next-step implications

Important distinctions:

### Report 3
- exploratory
- broader initial sweep
- discovered the phenomenon
- patching pilot was tiny

### Report 6
- focused
- design-driven
- verifies behavior/tracing
- runs structured small patching grid
- shows current patching does not restore refusal

---

## Step 6 — Hypothesis-by-hypothesis status

Create:

`## Current hypothesis status after Report 6`

Use the hypotheses from Report 5 / Report 6 design.

Likely hypotheses:

### H1 — Late-layer negative shift under prefilling
Status: supported if tracing shows late layers shift negative.

### H2 — Layer specificity
Status: supported if layers `24/27` shift more than `16/20`.

### H3 — Behavioral association
Status: partially supported or needs more direct analysis, depending on prompt-level label vs projection analysis.

### H4 — Patching restoration
Status: not supported under current intervention design.

### H5 — Late-layer patching more effective than comparison layer
Status: not supported if no late-layer recovery; maybe late-layer patching slightly worsens refusal.

For each hypothesis, include:
- status
- evidence
- caveat
- implication for Report 7

Do not overclaim.

---

## Step 7 — Validity and interpretation cautions

Create:

`## Interpretation cautions`

Include at least:

### Internal validity
- phrase-matching refusal classifier may mislabel borderline outputs
- patching source position may already contain attacked / negative signal
- null patching result may be due to intervention design rather than absence of causal role
- small prompt subset for patching
- summary mismatches if averaging over different token-position windows

### External validity
- one model
- one attack family
- limited prompt set
- may not generalize to suffix attacks, other models, or other refusal mechanisms

### Construct validity
- refusal-direction projection is a useful proxy, but not identical to “safety”
- refusal label is a behavior proxy, not a complete harmfulness/safety evaluation

---

## Step 8 — Report 7 plan

Create:

`## What Report 7 should improve`

Be specific.

Possible Report 7 improvements:

1. **Better patch source choice**
   - instead of source position `-1` under attacked condition if it is already negative,
   - try source from baseline `k=0`,
   - or source from an earlier more refusal-like token/position,
   - or explicitly add refusal-direction vector rather than copy attacked source component.

2. **Expanded patching only where useful**
   - focus on layer `24` and `27`
   - do not expand all layers
   - increase prompt count for best conditions

3. **Prompt-level association analysis**
   - compare projection values for refused vs complied outputs
   - test whether more negative projection predicts compliance

4. **Clarify generated-token vs all-token summaries**
   - standardize the window used for headline tables

5. **Optional k=10 / threshold analysis**
   - only if needed to support the plateau or threshold story

6. **Do not add suffix attacks unless everything else is done**
   - suffix attacks are out of scope unless time remains

---

## Step 9 — Create writeup-ready tables

If data exists, create markdown tables in `REPORT3_REPORT6_ANALYSIS.md` for:

### Behavioral summary
Columns:
- condition
- valid prompts
- refusals
- refusal rate
- errors

### Tracing summary by layer
Columns:
- condition
- layer
- mean projection
- notes

### Patching summary
Columns:
- layer
- source position
- target position
- baseline refusal rate
- patched refusal rate
- delta
- restored count
- lost count
- interpretation

If exact column names differ, adapt.

---

## Step 10 — Optional CSV exports

If useful, create a small folder:

`outputs/report6/analysis_for_writeup/`

and save:

- `report3_report6_behavior_comparison.csv`
- `report6_key_tracing_summary.csv`
- `report6_key_patching_summary.csv`

Only do this if the required data exists.

---

## Final deliverable

At the end, print a concise console summary:

### COMPLETE
- what was successfully summarized

### IMPORTANT FINDINGS
- top 3–5 results

### CAUTIONS
- top 3 caveats

### REPORT 7 PRIORITIES
- top 3 next improvements

### READY FOR REPORT 6 WRITING?
Answer clearly:
- yes
- yes but mention missing X
- no because Y

---

## Writing style for the generated document

Use clear language. Avoid hype.

Preferred phrasing:
- “This supports…”
- “This weakens…”
- “This does not rule out…”
- “Under the current intervention setting…”
- “The current evidence suggests…”

Avoid:
- “proves”
- “demonstrates mechanism conclusively”
- “disproves the hypothesis”
- “solves the project”

---

Start by inspecting the repo. Build the inventory first. Then generate the markdown document.
