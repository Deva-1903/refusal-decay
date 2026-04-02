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
