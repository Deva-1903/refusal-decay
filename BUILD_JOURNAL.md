# Build Journal

## What Changed

- Added a Report 6-specific experiment path under `configs/experiments/report6/`.
- Added three explicit Report 6 runner scripts:
  - `scripts/run_report6_generation.py`
  - `scripts/run_report6_tracing.py`
  - `scripts/run_report6_patching.py`
- Added two report-prep scripts:
  - `scripts/summarize_report6_results.py`
  - `scripts/plot_report6_results.py`
- Added `RUN_REPORT6.md` with command order, expected outputs, Unity notes, and resume guidance.
- Added `slurm/report6_pipeline.sh` for a one-job Unity run.
- Fixed input-device placement in generation and patching to avoid CPU/GPU mismatch warnings.
- Tightened patching internals so baseline generations and source components are cached across repeated patch conditions.
- Corrected `configs/model_8b.yaml` so it actually documents and points to the 8B setup.

## Existing Files Extended

- `src/generation/generator.py`
  - moved generated inputs onto the model input device before `generate()`
- `src/patching/patch.py`
  - moved inputs onto the model input device
  - cached baseline generations and source components
  - recorded `prefix_k`, `source_projection`, and `patch_applied`
- `configs/model_8b.yaml`
  - fixed comments and layer list

## New Files

- `configs/experiments/report6/generation_report6.yaml`
- `configs/experiments/report6/baseline_k0.yaml`
- `configs/experiments/report6/prefill_k3.yaml`
- `configs/experiments/report6/prefill_k10.yaml`
- `configs/experiments/report6/extract_direction.yaml`
- `configs/experiments/report6/trace_report6.yaml`
- `configs/experiments/report6/patch_report6.yaml`
- `scripts/run_report6_generation.py`
- `scripts/run_report6_tracing.py`
- `scripts/run_report6_patching.py`
- `scripts/summarize_report6_results.py`
- `scripts/plot_report6_results.py`
- `slurm/report6_pipeline.sh`
- `RUN_REPORT6.md`
- `BUILD_JOURNAL.md`

## Why These Changes

- Report 6 needs a narrow, clean pipeline rather than the broader exploratory Report 3 structure.
- The repo already had the core mechanics, so the work focused on orchestration, stable output locations, report tables, and report-ready figures.
- The new scripts avoid replacing the existing pipeline. They mostly reuse the current generation, direction, tracing, and patching modules.

## Assumptions I Had To Make

- The main observational tracing block should remain harmful-only, with benign prompts used for baseline behavior and direction extraction rather than full tracing.
- `k=10` should stay supported because it was already cheap to keep in the config surface, but it is optional in the runbook.
- A small patching subset of `10` harmful prompts is a reasonable default for Unity runtime.
- The prompt-level raw-data figure should be chosen automatically from the prompt showing the largest mean layer-27 drop from harmful `k=0` to harmful `k=3`.

## Known Limitations

- The refusal labeler is still phrase-based. It is centralized and easy to inspect, but it is still heuristic.
- Patching remains same-prompt self-patching, not cross-prompt patching.
- The tracing summaries and figures assume the Report 6 condition naming scheme like `harmful_k03`.
- The new plotting script depends on the summary outputs and the combined trace file.
- The batch script is a template and still needs Unity-specific partition and environment edits before submission.

## Practical TODOs If You Extend This Later

- Add a guarded or manual-validation label pass for ambiguous refusal/compliance cases.
- Add an optional benign tracing config if you later want a stronger internal control block.
- Add a dedicated patching config for a second source-position comparison if Report 6 follow-up time allows.
