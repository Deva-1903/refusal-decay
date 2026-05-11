# Build Journal Report 7

## What Changed

- Added Report 7 configs under `configs/experiments/report7/`.
- Added cross-condition baseline-source patching.
- Added additive refusal-direction intervention with random-direction control.
- Added prompt-level projection/label association analysis.
- Added standardized tracing summaries.
- Added Report 6 vs Report 7 intervention comparison.
- Added Report 7 plotting and verification scripts.
- Added Report 7 run and experiment-plan docs.
- Added a runbook shortcut for reusing the verified Report 6 refusal direction when the model/layers match.
- Tightened the verifier's next-step list so it prioritizes whichever Report 7 artifacts are actually missing.

## New Files

- `configs/experiments/report7/generation_report7.yaml`
- `configs/experiments/report7/tracing_report7.yaml`
- `configs/experiments/report7/patching_report7.yaml`
- `configs/experiments/report7/additive_intervention_report7.yaml`
- `src/patching/report7_interventions.py`
- `scripts/run_report7_generation.py`
- `scripts/run_report7_cross_condition_patching.py`
- `scripts/run_report7_additive_direction_intervention.py`
- `scripts/summarize_report7_tracing.py`
- `scripts/analyze_report7_prompt_association.py`
- `scripts/compare_report6_report7_patching.py`
- `scripts/plot_report7_results.py`
- `scripts/verify_report7_outputs.py`
- `REPORT7_EXPERIMENT_PLAN.md`
- `RUN_REPORT7.md`
- `VERIFY_REPORT7.md`
- `BUILD_JOURNAL_REPORT7.md`

## Why

Report 6 had a strong observational result but a weak causal test because the patch source came from the attacked condition. Report 7 adds two focused tests that address that weakness:

- Cross-condition patching uses a cleaner baseline source state.
- Additive intervention tests whether directly increasing the refusal-direction component is sufficient under attack.

## Commands Tested

- `python -m py_compile src/patching/report7_interventions.py scripts/run_report7_generation.py scripts/run_report7_cross_condition_patching.py scripts/run_report7_additive_direction_intervention.py scripts/summarize_report7_tracing.py scripts/analyze_report7_prompt_association.py scripts/compare_report6_report7_patching.py scripts/plot_report7_results.py scripts/verify_report7_outputs.py`
- `pytest -q`
- `python scripts/run_report7_generation.py --config configs/experiments/report7/generation_report7.yaml --conditions harmful_k00 harmful_k03 benign_k00`
- `python scripts/summarize_report7_tracing.py`
- `python scripts/analyze_report7_prompt_association.py`
- `python scripts/compare_report6_report7_patching.py`
- `python scripts/plot_report7_results.py`
- `python scripts/verify_report7_outputs.py`

## Output Files Produced Locally

Generated from valid Report 6 outputs without running a GPU model:

- `outputs/report7/generations/report7_generation_summary.csv`
- `outputs/report7/generations/report7_generation_combined.jsonl`
- `outputs/report7/summaries/generation_refusal_rates_by_condition.csv`
- `outputs/report7/summaries/generation_prompt_labels.csv`
- `outputs/report7/summaries/trace_generated_token_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_all_position_mean_by_condition_layer.csv`
- `outputs/report7/summaries/trace_token_trajectory_by_condition_layer.csv`
- `outputs/report7/summaries/prompt_projection_label_association.csv`
- `outputs/report7/summaries/prompt_projection_label_differences.csv`
- `outputs/report7/summaries/report6_vs_report7_intervention_comparison.csv`
- `outputs/report7/plots/report7_refusal_rate_by_condition.png`
- `outputs/report7/plots/report7_generated_mean_projection_by_layer.png`
- `outputs/report7/plots/report7_layer27_projection_trajectory.png`
- `outputs/report7/plots/report7_layer24_projection_trajectory.png`
- `outputs/report7/plots/report7_layer27_projection_by_label.png`
- `outputs/report7/plots/report7_report6_vs_report7_intervention_comparison.png`
- `VERIFY_REPORT7.md`

Still to produce on Unity:

- `outputs/report7/patching/cross_condition_patching_results.csv`
- `outputs/report7/patching/cross_condition_patching_summary.csv`
- `outputs/report7/interventions/additive_direction_results.csv`
- `outputs/report7/interventions/additive_direction_summary.csv`

## Known Limitations

- The refusal labeler remains phrase-based.
- Cross-condition patching is same-prompt across conditions, not cross-prompt.
- Additive intervention uses a simple unit-direction alpha scale.
- Random direction is a lightweight negative control, not a full causal-control suite.
- Runtime defaults are intentionally small.
- The local verifier currently reports the new intervention outputs as missing, because those require a Unity GPU run.

## Deferred

- Larger prompt count for best settings.
- Layer `20` intervention grid.
- Target position `7`.
- Orthogonal direction control if random control is insufficient.

## Cleanup Note

Cleaned stale locally generated Report 7 outputs before running real Unity GPU experiments.
