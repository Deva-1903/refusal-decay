# Research Log — Refusal Decay

Tracks initial assumptions, setup decisions, failures, fixes, and scope changes.
Update this file whenever something non-obvious is learned or changed.

Format:
```
## [Date] Short description
**What happened**: ...
**Why**: ...
**Decision / Fix**: ...
**Impact on scope**: ...
```

---

## [2026-04-02] Initial codebase setup

**What happened**: Built the initial research infrastructure from scratch.

**Assumptions made at setup**:
1. Llama-3.2-3B-Instruct will be used as the primary model (28 layers, hidden size 3072).
2. Prefilling attacks are implemented by appending k tokenized prefix tokens to the chat-formatted input before calling `model.generate()`.
3. The refusal direction is extracted via difference-in-means (DiM) at the last token position of the prompt. This is the standard approach from Representation Engineering (Zou et al. 2023) and Arditi et al. (2024).
4. Activation hooks (PyTorch `register_forward_hook`) are used for tracing — simpler and more reliable than wrapping the generation loop.
5. During autoregressive generation, the prefill step has `seq_len > 1`; subsequent steps have `seq_len = 1`. The tracing and patching hooks use this to distinguish steps.
6. The fake dataset (25 harmful + 25 benign prompts) is sufficient to test the pipeline end-to-end. Real data (AdvBench, HarmBench) will be dropped in later.
7. Greedy decoding (`do_sample=False`) is used throughout for determinism.

**Setup choices**:
- Config system: YAML + dot-access `Config` class. Chose this over pydantic to minimize dependencies and keep the code readable.
- Output format: JSONL for text, Parquet for tidy trace tables (efficient columnar reads for analysis).
- All scripts are resume-safe: check for existing output files before running.
- Patching implemented with hooks (not pyvene) at this stage for simplicity. pyvene is listed in requirements for future use.
- `direction_position: -1` (last token of prompt) for DiM. This matches standard practice, but may need to be changed if prompt endings are too similar across categories.

**Known limitations at setup**:
- The phrase-list refusal classifier is a heuristic and will have false positives/negatives.
- The patching experiment patches within the same prompt (self-patching), not across prompts. Cross-prompt patching is a TODO.
- Hook-based activation collection during `generate()` stores ALL intermediate states; this could OOM for long sequences. Mitigated by limiting `max_new_tokens` during tracing.

**Impact on scope**: None — this is the initial setup. All scope decisions documented in README.

---

---

## [2026-04-02] Corrected model target, patching direction, classifier schema, and data labeling

### Fix 1: 8B is the research target, 3B is the smoke-test model

**What happened**: Initial setup defaulted to Llama-3.2-3B-Instruct throughout all configs and documentation, treating it as the primary model.

**Why this was wrong**: The course project is scoped to Llama-3.1-8B-Instruct. Using 3B as the default would produce results that don't match the project framing. 3B has only 28 layers vs 32 for 8B, and its refusal behavior and representation geometry may differ.

**Decision / Fix**: Changed `configs/base.yaml` to default to 8B (`meta-llama/Llama-3.1-8B-Instruct`). Updated layer list from `[0,4,8,12,16,20,24,27]` to `[0,4,8,12,16,20,24,28,31]`. Added `configs/model_3b.yaml` as an explicit smoke-test override with a clear comment. README first-24-hours plan now says: smoke-test on 3B first, then rerun on 8B for actual results.

**Impact on scope**: No change to experimental design. Only model default and documentation changed.

---

### Fix 2: Patching success criterion was backwards

**What happened**: Initial code and README implied the patching experiment succeeds when cases flip from "refusal" to "compliance" — i.e., patching destroys refusal. The README asked "do any cases flip from refusal to compliance?"

**Why this was wrong**: The research hypothesis is that the refusal direction at early token positions *causally sustains* refusal behavior. Patching the early-token refusal signal INTO later positions should therefore *increase* refusal probability (compliance → refusal), not decrease it. The wrong success criterion would have led to misinterpretation of the results.

**Decision / Fix**:
- Updated `src/patching/patch.py` docstring to state the correct hypothesis and expected direction of effect.
- Rewrote the summary logic in `scripts/run_patching.py` to track three cases: `refusal_restored` (compliance→refusal, expected positive effect), `refusal_lost` (refusal→compliance, unexpected), and no change.
- Renamed output fields to `baseline_phrase_label` / `patched_phrase_label` to avoid confusion with future guard-based labels.
- Updated `plot_results.py` to annotate that "↑ patched = expected causal effect".

**Impact on scope**: No change to what is computed — only the interpretation and summary reporting is fixed.

---

### Fix 3: Llama Guard was misleadingly implied to be available

**What happened**: The initial classifier code had a commented-out stub described as "TODO", and the config had `use_llama_guard: false` with a model name set. This implied guard-based evaluation was nearly ready.

**Why this was wrong**: Llama Guard integration requires a separate model download, a conversation template, and a proper pipeline. None of this was implemented. Keeping the stub ambiguous risked over-reporting evaluation quality.

**Decision / Fix**:
- Completely rewrote `src/classification/refusal_classifier.py` to define a proper `GuardClassifier` abstract interface (ABC) with a docstring stating "NOT YET IMPLEMENTED".
- Added `classify_responses()` parameter `guard_clf=None` with a warning if passed (since it cannot be used yet).
- Output schema now explicitly reserves `safety_guard_label`, `guard_model_name`, `disagreement_flag` as `null` fields — honest about what is missing.
- Set `llama_guard_model: null` in config with a comment stating it is not integrated.
- README now has a dedicated section explaining what the phrase-list classifier does and does not guarantee.

**Impact on scope**: Evaluation is still phrase-list only for now. Future integration requires implementing `LlamaGuardClassifier(GuardClassifier)`.

---

### Fix 4: Fake prompt data was not clearly labeled

**What happened**: Placeholder prompts in `data/harmful_prompts.jsonl` and `data/benign_prompts.jsonl` had `"source": "fake_example"`, which is ambiguous.

**Why this was wrong**: Any output artifact built on this data would not be clearly flagged as using synthetic prompts, making it easy to accidentally treat results as if they came from real benchmarks.

**Decision / Fix**:
- Changed `"source": "fake_example"` → `"source": "synthetic_placeholder"` in all 50 placeholder records.
- Created `data/README.md` with clear instructions for obtaining AdvBench (harmful) and Alpaca (benign) data, plus the expected JSONL schema.
- Created `scripts/normalize_prompts.py` to convert CSV/JSONL from external sources into the repo schema.
- README now has a "Before Running Experiments" section that makes replacing placeholder data a required step.

**Impact on scope**: No change to pipeline logic. All downstream code works identically with real data once the JSONL files are replaced.

---

## [Future entries — template]

## [YYYY-MM-DD] <Short description>

**What happened**: 

**Why**: 

**Decision / Fix**: 

**Impact on scope**: 

---

## Questions / Open Issues

Track unresolved questions here:

1. **Does the last-token position for DiM give a clean direction?** The last token of a harmful prompt vs benign prompt may differ structurally (punctuation, stop words). May need to use the mean over all tokens, or a specific content token.

2. **Will k=1 meaningfully differ from k=0?** A single "Sure" token may not strongly bias the continuation. The sweep will tell.

3. **Is the refusal direction truly 1-dimensional?** Arditi et al. found one dominant direction, but this may vary by model version. A PCA check on the DiM activations would be informative.

4. **Patching step vs. token position alignment**: The step counter in the patching hook is relative to autoregressive steps (0 = first generated token). But `target_position` in config is also 0-indexed from generation. This is consistent but needs verification with actual model outputs.

5. **Multi-GPU hooks**: If `device_map="auto"` splits layers across GPUs, the hooks need tensors on the right device. The current code calls `.detach().cpu()` which should handle this, but test carefully.
