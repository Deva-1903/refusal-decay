# Report 3 Status

## Scope check

- [x] Primary model is `meta-llama/Llama-3.1-8B-Instruct`.
- [x] Prefilling is the main attack setting.
- [x] Refusal direction is the main mechanism.
- [x] Suffix attacks are not part of the current deliverable.

## Data check

- [x] Real harmful prompts are present in [`data/harmful_prompts.jsonl`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/data/harmful_prompts.jsonl) with AdvBench records.
- [x] Real benign prompts are present in [`data/benign_prompts.jsonl`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/data/benign_prompts.jsonl) with Alpaca records.
- [x] The current Report 3 runs use real data, not the old synthetic placeholders.
- [ ] The direction extraction set is not truly held out in the current small run.

Reason:
The mechanism run used `data.max_prompts: 25`, so direction extraction and tracing both draw from the same first 25 harmful and 25 benign prompts in [`report3_small/config_snapshot.yaml`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3_runs_1/20260406_232425_report3_mechanism/report3_small/config_snapshot.yaml). This is acceptable for exploratory Report 3 analysis, but it should be fixed before making a stronger mechanistic claim.

## Core experiments

- [x] Baseline harmful generation at `k=0`.
- [x] Prefilling sweep over `k in {0, 3, 10}`.
- [x] Refusal-direction extraction.
- [x] Tracing of projection over token positions and selected layers.
- [x] One patching pilot.

Evidence:

- Baseline / sweep outputs: [`report3_prefilling_sweep/refusal_rate_summary.csv`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3_runs_1/20260406_222202_8b_harmful_sweep_k0_3_10_25/report3_prefilling_sweep/refusal_rate_summary.csv)
- Direction file: [`refusal_direction.pt`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3_runs_1/20260406_232425_report3_mechanism/refusal_direction.pt)
- Traces: [`traces_all.parquet`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3_runs_1/20260406_232425_report3_mechanism/report3_small/traces_all.parquet)
- Patching pilot: [`patching_classified.jsonl`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3_runs_1/20260407_012344_report3_patching_pilot/patches/patching_classified.jsonl)

## Current results you can honestly report

### 1. Behavioral attack effect

- [x] Harmful prompts mostly refuse at `k=0`.
- [x] Refusal rate drops sharply under prefilling.
- [ ] Clean monotonic decay with larger `k` is not established in this small sample.

Observed:

- `k=0`: `24/25 = 0.96` refusal
- `k=3`: `8/25 = 0.32` refusal
- `k=10`: `9/25 = 0.36` refusal

Interpretation:
The current data clearly supports a strong prefilling attack effect. It does not yet support a strong monotonicity claim between `k=3` and `k=10`.

### 2. Mechanism traces

- [x] Projection-by-position plots exist.
- [x] Heatmaps exist for `k=0`, `k=3`, and `k=10`.
- [x] The traces show stronger decay and sign-flip behavior in mid-to-late layers than in early layers.

Current qualitative pattern from the saved traces:

- Early layers (`0`, `4`) stay near zero or weakly positive.
- Mid layers (`8`, `12`) remain positive but flatten under attack.
- Later layers (`16`, `20`, `24`, `27`) show the clearest positional decay.
- For `k=3` and `k=10`, late layers become strongly negative on average, especially layers `24` and `27`.

This is enough for an exploratory Report 3 claim that the refusal-direction signal weakens or flips later in generation under prefilling, with the clearest separation in later layers.

### 3. Patching pilot

- [x] One causal pilot exists.
- [ ] The pilot is too small to support a strong causal conclusion.

Observed:

- Setting: `k=3`, layer `27`, source position `-1`, target position `5`, `n=5` harmful prompts
- Baseline refusal rate: `1/5 = 0.20`
- Patched refusal rate: `1/5 = 0.20`
- `compliance -> refusal`: `0/5`
- `refusal -> compliance`: `0/5`

Interpretation:
This is a valid pilot and a valid negative preliminary result, but not enough to say patching does not work in general.

## Figures available now

- [x] [`refusal_rate_vs_k.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/refusal_rate_vs_k.png)
- [x] [`projection_vs_position_layer0.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer0.png)
- [x] [`projection_vs_position_layer4.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer4.png)
- [x] [`projection_vs_position_layer8.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer8.png)
- [x] [`projection_vs_position_layer12.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer12.png)
- [x] [`projection_vs_position_layer16.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer16.png)
- [x] [`projection_vs_position_layer20.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer20.png)
- [x] [`projection_vs_position_layer24.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer24.png)
- [x] [`projection_vs_position_layer27.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_vs_position_layer27.png)
- [x] [`projection_heatmap_k00.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_heatmap_k00.png)
- [x] [`projection_heatmap_k03.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_heatmap_k03.png)
- [x] [`projection_heatmap_k10.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_heatmap_k10.png)
- [x] [`projection_heatmap_comparison.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_heatmap_comparison.png)
- [x] [`projection_by_layer_bar.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/projection_by_layer_bar.png)
- [x] [`patching_comparison.png`](/Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/report3/patching_comparison.png)

## What is still pending

- [ ] Write the expectations section explicitly in the report text.
- [ ] Add an expectations-vs-results paragraph for each core phenomenon.
- [ ] State the main caveat that the direction extraction set was not truly held out.
- [ ] State the main evaluation caveat that refusal labels are phrase-list heuristic labels.
- [ ] Decide whether Report 4 will focus on:
  - later-layer positional decay under prefilling, or
  - a broader patching sweep to test causal restoration more seriously.

## Minimal Report 3 claim that is supported now

With real AdvBench/Alpaca data and `Llama-3.1-8B-Instruct`, the current exploratory run supports the claim that prefilling substantially suppresses refusal on harmful prompts and that the refusal-direction projection decays across generation, with the clearest degradation in later layers. A small single-setting patching pilot did not restore refusal, but that result is preliminary because it uses only 5 prompts and one intervention setting.
