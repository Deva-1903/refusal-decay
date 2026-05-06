# Claude Code prompt — Verify Report 6 experiments and generate a writeup-ready summary

Use this prompt inside Claude Code at the root of my current project repo.

---

You are helping me verify my **COMPSCI 602 Project Report 6** experiment pipeline and outputs before I start writing the report.

## Project context

The project is about **prefilling attacks on safety-aligned LLMs** and the **refusal-direction signal**.

### Current narrowed Report 6 goal
I am **not** trying to do the whole project at once.

For Report 6, the goal is to verify and summarize a **small, focused study**:

1. **Observational evidence**
   - Compare harmful prompts under:
     - baseline `k=0`
     - prefilling attack `k=3`
   - Optionally include `k=10` only if already available or cheap.
   - Main measurements:
     - refusal / compliance behavior
     - refusal-direction projection by layer and token position
   - Key layers:
     - comparison layers: `16`, `20`
     - main late layers: `24`, `27`

2. **Initial causal evidence**
   - Run / verify targeted patching under attacked condition `k=3`
   - Main target layers:
     - `24`, `27`
   - Comparison patch layer:
     - `16`
   - Main target token positions:
     - `1`, `3`, `5`
   - Main source position:
     - `-1` (last input / prefill token)
   - Smaller prompt subset is acceptable for patching if runtime is large.

### Intended causal question
For Report 6, the core question is:

**How and under what intervention conditions does the late-layer negative shift in the refusal-direction signal contribute to refusal failure under prefilling attacks?**

This is a **causal question** that is informative about mechanism.
Do not rewrite the project as if it is already a full mechanistic proof.

---

## What I want you to do

I want you to act like a careful research engineer and internal reviewer.

Your job is to:

1. **Inspect the repo first**
   - Understand the current structure
   - Identify existing scripts, configs, outputs, notebooks, logs, and helper code
   - Prefer reusing existing code paths
   - Do **not** create parallel duplicate pipelines unless necessary

2. **Verify whether the current experiments for Report 6 are actually complete**
   - Check whether the required runs exist
   - Check whether they finished successfully
   - Check whether outputs match expected schemas
   - Check whether plots / tables can be generated from the outputs
   - Check whether the experimental scope matches Report 6 rather than some older or broader project scope

3. **Generate a structured verification summary**
   - Tell me exactly:
     - what has been run
     - what is missing
     - what looks broken or suspicious
     - what is sufficient to begin writing Report 6
     - what should be deferred to Report 7

4. **If needed, add small missing utilities only**
   - Example:
     - summary scripts
     - validation scripts
     - figure-generation scripts
     - run manifests
   - But keep changes minimal and preserve the existing structure

5. **Produce a final markdown document**
   - Name it something like:
     - `VERIFY_REPORT6.md`
   - It should be written for me, not for a grader
   - It should help me decide:
     - “Can I start writing the report now?”
     - “What exact experiments still need to be run?”

---

## Very important constraints

### Scope control
Do **not** silently expand scope.
For Report 6, prefer this priority order:

#### Must-have for Report 6
- harmful baseline `k=0`
- harmful prefilling `k=3`
- benign baseline `k=0`
- refusal behavior summaries
- tracing summaries for layers `16, 20, 24, 27`
- at least one key late-layer plot
- at least one comparison-layer plot
- small patching study under `k=3`

#### Good-to-have but optional for Report 6
- harmful `k=10`
- more prompt-level raw plots
- extra summary tables

#### Better deferred to Report 7
- larger patch grids
- extra source positions
- much larger prompt counts for patching
- suffix attacks
- attention-head or neuron analyses
- broad factorial sweeps

### Be skeptical
If outputs look inconsistent, say so directly.
If logs suggest partial failure, say so directly.
If behavioral and tracing results do not line up, say so directly.

Do not flatter.
Do not assume success.
Do not invent missing evidence.

### Preserve reproducibility
If you add anything, also add:
- exact command(s) to run it
- expected output path(s)
- short description of what it checks

---

## What to inspect

Please inspect all relevant folders and infer the actual structure of this repo.
Likely useful things include:

- experiment configs
- generation outputs
- tracing outputs
- patching outputs
- logs
- notebooks
- plotting scripts
- helper utilities
- build journals / run docs
- report drafts if they exist

If there is a `RUN_REPORT6.md`, inspect it carefully and compare it against the actual outputs.
If there are Report 3-era outputs, distinguish them from actual Report 6 results.

---

## What to verify in detail

## A. Behavioral runs
Check whether these exist and look valid:

- harmful `k=0`
- harmful `k=3`
- benign `k=0`
- optional: harmful `k=10`

For each condition, verify:
- output file exists
- number of prompts processed
- whether outputs are complete
- whether refusal/compliance labeling exists
- whether summary refusal counts can be computed

If summary files already exist, verify they match raw outputs.

## B. Refusal-direction extraction
Check:
- whether refusal directions were extracted successfully
- which layers are available
- whether the saved artifact is compatible with the tracing/patching code

## C. Tracing runs
Check whether tracing exists for:
- harmful `k=0`
- harmful `k=3`
- optional `k=10`
- benign `k=0` if available

Verify:
- projection output schema
- prompt count
- layer coverage
- token-position coverage
- whether layers `16, 20, 24, 27` are available
- whether summary CSV/parquet files can be generated

Also inspect whether the actual results seem to show:
- stronger late-layer negative shift under prefilling
- weaker change in comparison layers

Do not overclaim — just note what the current outputs support.

## D. Patching runs
Check whether patching exists for:
- attacked `k=3`
- layers `16, 24, 27`
- target positions `1, 3, 5`
- source position `-1`
- prompt subset used

Verify:
- which combinations were actually run
- whether each run completed
- whether baseline attacked outputs and patched outputs are both present
- whether a summary effect table can be computed

Check whether patching shows:
- any refusal recovery at all
- stronger effect in late layers than layer `16`
- or no effect under current settings

Again, do not force a positive story.

## E. Figures / tables needed for Report 6
Tell me whether I already have enough material to write the report.

Minimum useful figures/tables:
1. refusal-rate table for baseline vs attack
2. refusal-rate plot vs `k` if available
3. heatmap or equivalent summary of projection by layer × token position
4. line plot for a key late layer (`24` or `27`)
5. line plot for a comparison layer (`16` or `20`)
6. patching summary table
7. optional prompt-level raw plot for one key layer

For each, tell me:
- already available
- can be generated from existing outputs
- missing because experiment missing
- missing because summary code missing

---

## What output I want from you

Please produce these deliverables.

### 1. Console summary for me
At the end, print a concise summary with sections:

- `COMPLETE`
- `INCOMPLETE`
- `SUSPICIOUS / NEEDS REVIEW`
- `READY TO WRITE?`
- `RUN THESE NEXT`

### 2. Markdown file
Create `VERIFY_REPORT6.md` with the following structure:

#### `# Report 6 verification summary`
#### `## 1. Current project scope for Report 6`
Short reminder of what we are trying to do and what is out of scope.

#### `## 2. What was found in the repo`
Repo structure, key scripts/configs/outputs.

#### `## 3. Behavioral experiments`
What exists, what is missing, what results are currently supported.

#### `## 4. Tracing experiments`
What exists, what is missing, what results are currently supported.

#### `## 5. Patching experiments`
What exists, what is missing, what results are currently supported.

#### `## 6. Figures and tables available for the report`
Mark each as:
- available now
- derivable now
- missing

#### `## 7. Can I start writing Report 6?`
Give a blunt answer:
- yes
- yes, but run X first
- no, because Y is missing

#### `## 8. Exact next commands to run`
Only include commands that are actually needed.

#### `## 9. What should be deferred to Report 7`
Be explicit.

### 3. Optional helper scripts
Only if genuinely needed, add small scripts such as:
- `scripts/verify_report6_outputs.py`
- `scripts/summarize_report6_behavior.py`
- `scripts/summarize_report6_tracing.py`
- `scripts/summarize_report6_patching.py`

If you add these, document them in `VERIFY_REPORT6.md`.

---

## Preferred style

- Be precise
- Be skeptical
- Be practical
- Optimize for getting to a writeable Report 6
- Do not bloat the scope
- Do not fabricate missing evidence
- If something is ambiguous, say what file/log/output must be checked next

---

## Final decision rule

At the end, I want a direct recommendation:

- **START WRITING NOW**
or
- **RUN THESE 1–3 THINGS FIRST**

Do not give me a vague answer.

Start by inspecting the repo and building an inventory before making any changes.
