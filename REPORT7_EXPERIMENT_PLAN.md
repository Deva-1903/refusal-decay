# Report 7 Experiment Plan

## Guiding Question

When prefilling attacks push the late-layer refusal-direction signal negative, can targeted interventions that restore or add a cleaner refusal-direction component recover refusal behavior?

This is a causal question that informs mechanism. It is not a claim that the project fully explains refusal behavior.

## Hypotheses

- H1: Prefilling at `k=3` reduces refusal on harmful prompts compared with baseline `k=0`.
- H2: Refusal-direction projection shifts more strongly negative in late layers `24` and `27` than comparison layers `16` and `20`.
- H3: Among attacked harmful prompts, complied prompts have more negative late-layer projection than refused prompts.
- H4: Cross-condition patching from baseline `k=0` source activations into attacked `k=3` target runs restores refusal more often than Report 6 same-condition attacked-source patching.
- H5: Adding a scaled refusal-direction component directly into attacked residual states increases refusal if the direction is causally sufficient under those settings.
- H6: Effects should be stronger in late layers and with the real refusal direction than in comparison/control settings.

## Design

Report 7 keeps the Report 6 behavioral and tracing structure, then adds two focused causal tests:

- Cross-condition source patching: copy the refusal-direction component from the same prompt under `harmful_k00` into the attacked `harmful_k03` generation.
- Additive direction intervention: add `alpha * refusal_direction[layer]` directly into attacked `harmful_k03` generation.

Default intervention grid:

- Prompts: first `10` harmful prompts
- Layers: `16`, `24`, `27`
- Target positions: `0`, `1`, `3`, `5`
- Cross-condition source position: `-1` from baseline `k=0`
- Additive alpha values: `0.5`, `1.0`, `2.0`
- Additive controls: real refusal direction plus random normalized direction

## Scope Boundaries

Implemented for Report 7:

- Behavioral verification
- Tracing summaries
- Prompt-level label/projection association
- Cross-condition baseline-source patching
- Additive refusal-direction intervention
- One negative-control direction for additive intervention
- Report 6 vs Report 7 intervention comparison

Deferred:

- Suffix attacks
- Attention heads
- Neurons
- All-layer sweeps
- Multi-model runs
- Training or fine-tuning

## Interpretation Rules

If baseline-source patching restores refusal, interpret it as evidence that Report 6 same-condition patching may have used an already-corrupted source state.

If additive intervention restores refusal, interpret it as stronger causal evidence that increasing the refusal-direction component can recover refusal under some attacked settings.

If neither intervention restores refusal, interpret the late-layer negative shift as robustly associated with attack success but insufficient by simple one-direction restoration alone.

Use careful language: supports, weakens, suggests, consistent with, and under this intervention setting.

Avoid language like proves, fully explains, disproves, or solves safety alignment.
