# Verify Report 7

## COMPLETE

- Behavioral: harmful_k00: 25 rows, 0 errors
- Behavioral: harmful_k03: 25 rows, 0 errors
- Behavioral: benign_k00: 25 rows, 0 errors
- Tracing: Tracing summary covers layers 16/20/24/27 and harmful_k00/harmful_k03.
- Prompt-Level Association: Prompt association includes layers 24/27 and both label groups.

## MISSING

- Cross-Condition Patching: Missing cross-condition patching results: /Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/outputs/report7/patching/cross_condition_patching_results.csv
- Additive Intervention: Missing additive intervention results: /Users/devaanand/Documents/Projects/Coding Stuff/refusal-decay/outputs/report7/interventions/additive_direction_results.csv

## SUSPICIOUS

- None

## READY TO WRITE REPORT 7?

Not yet

## RUN THESE NEXT

- `python scripts/run_report7_cross_condition_patching.py --config configs/experiments/report7/patching_report7.yaml --direction-path outputs/report7/directions/refusal_direction.pt`
- `python scripts/run_report7_additive_direction_intervention.py --config configs/experiments/report7/additive_intervention_report7.yaml --direction-path outputs/report7/directions/refusal_direction.pt`
- `python scripts/compare_report6_report7_patching.py`
- `python scripts/plot_report7_results.py`
- `python scripts/verify_report7_outputs.py`
