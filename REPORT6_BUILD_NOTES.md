# Report 6 — Build Notes

Build timestamp: 2026-05-06 (post-feedback revision).
Compiled PDF: `report6/report6.pdf` (7 pages, letter).
LaTeX source: `report6/report6.tex` plus `report6/compsci602-project.sty` and `report6/references.bib`.

## Post-feedback fixes (2026-05-06, second pass)

1. **Footer.** `compsci602-project.sty` shipped with `COMPSCI 602, Fall 2024, Project Report 6` in the first-page notice string. Edited in place to `COMPSCI 602, Spring 2026, Project Report 6`. Confirmed visible at the bottom of page 1 of the rebuilt PDF.
2. **Section 6 layout.** In the first build, Section 6 (`Updated Hypothesis Status`) appeared with no nearby table — the hypothesis-status table was queued behind the Section 5 patching figure and floated past Section 7. Fix: added `\usepackage{placeins}` and a single `\FloatBarrier` immediately before Section 6, plus shrank Figure 4 (`report6_patching_comparison.png`) from `0.85\linewidth` to `0.7\linewidth` and changed its placement from `[h]` to `[!t]`. Result: all Section 5 floats (Figure 3, Table 3, Figure 4) are flushed before Section 6, and Table 4 sits on the same page as the Section 6 header.
3. **Layer-24 source sentence.** Removed the comparison to "what a $k=0$ forward pass would provide" because Section 4 only reports a generated-token mean for $k=0$, not a $k=0$ last-input-token activation. Replaced with: "At layer 24, the source projection is positive ($+0.948$), but this intervention still produced no restoration in any cell. This suggests that source positivity alone is not sufficient under the current same-condition patching design; testing whether a clean $k=0$ source activation would behave differently is left to Report 7."
4. **Tracing scope wording.** Removed the implication that benign Alpaca prompts were traced. Section 2 now says: "For tracing, I analyze 25 harmful AdvBench prompts under $k\in\{0, 3, 10\}$. ... The benign Alpaca set is used only as a behavioral control (Section~\ref{sec:behavior}); benign tracing is not part of Report 6."
5. **Bib + conclusion compaction.** Switched `\bibliographystyle` from `plain` to `abbrv` and wrapped the bibliography in `{\small ...}`; tightened one paragraph in Section 7.1. This brought the final report from 8 pages back down to 7.

## What goes into the report

### Source CSVs (verified, used as-is)

| Section | Source file | Used for |
|---|---|---|
| Behavioral table + bar chart | `outputs/report6/summaries/generation_refusal_rates_by_condition.csv` | harmful_k00 23/25=0.92, harmful_k03 8/25=0.32, benign_k00 0/25=0.00 |
| Behavioral k=10 reference | `outputs/report6/analysis_for_writeup/report3_report6_behavior_comparison.csv` (R3 pilot row) | 9/25=0.36 — labeled as Report 3 reference, not main R6 result |
| Tracing table | `outputs/report6/summaries/trace_mean_projection_by_condition_layer.csv` | layer 16/20/24/27 means, generated-tokens-only window |
| Patching table | `outputs/report6/summaries/patching_refusal_recovery_summary.csv` | 9 cells, baseline/patched rates, restored/lost counts, mean source projection |
| Cross-check | `outputs/report6/analysis_for_writeup/report6_key_tracing_summary.csv`, `report6_key_patching_summary.csv` | confirms the same numbers per layer/cell |

### Figures embedded
All figures are copied (unedited) from `outputs/report6/plots/` into `report6/figures/`:

- `report6_refusal_rate_by_condition.png` — Figure 1 (3-condition bar chart; valid since the May 6 19:38 regeneration after the L40S generation rerun).
- `report6_heatmap_baseline_vs_prefill.png` — Figure 2 (layer × token-position heatmap, 3 conditions).
- `report6_projection_vs_token_layer27.png` and `report6_projection_vs_token_layer16.png` — Figure 3 (layer-27 vs layer-16 line plot pair).
- `report6_patching_comparison.png` — Figure 4 (attacked-no-patch vs patched bar chart over the 9 cells).

`report6_prompt_level_layer27_advbench_0019.png` is *not* embedded; it would only be needed if the report did a single-prompt deep dive, which this draft doesn't.

### Documents read for framing
- `REPORT3_REPORT6_ANALYSIS.md` — the canonical evidence document; this is what the corrections to VERIFY_REPORT6 are based on.
- `VERIFY_REPORT6.md` — older audit; superseded for the behavioral generation status.
- `RUN_REPORT6.md`, `README.md`, `RESEARCH_LOG.md` — pipeline and scope context.
- Report 3 PDF (`602_Project_Report3.pdf`) — for prior-results framing.
- Report 4 PDF (`602_Project_Proposal_4 (2) (1).pdf`) — for the original RQ wording the professor asked us to revise.
- Report 5 PDF (`report5_revised (1).pdf`) — for the narrowed design we are now executing.

## Numbers used (sanity checklist — every number in the report appears below)

Behavioral (R6 pipeline):
- harmful_k00: n=25 valid, 23 refusals, refusal rate 0.92.
- harmful_k03: n=25 valid, 8 refusals, refusal rate 0.32.
- benign_k00: n=25 valid, 0 refusals, refusal rate 0.00.
- harmful_k10: R6 generation failed (GTX 1080 Ti / sm_61). Cited from R3 pilot only: 9/25=0.36.

Tracing (generated-tokens-only mean projection, n_prompts=25 for every condition; n_records 403 / 1124 / 1046 for k=0/3/10):
- layer 16: +1.271 (k=0), +0.172 (k=3), -0.270 (k=10)
- layer 20: +1.612 (k=0), -0.374 (k=3), -0.822 (k=10)
- layer 24: +0.665 (k=0), -3.110 (k=3), -3.463 (k=10)
- layer 27: -0.376 (k=0), -5.285 (k=3), -5.811 (k=10)
- Δ(k=3 − k=0): −1.099, −1.986, −3.775, −4.909
- Δ(k=10 − k=0): −1.541, −2.434, −4.128, −5.435

Patching (harmful_k03, n=10 prompts/cell, baseline subset refusal rate 0.50):
- L16, source proj +0.543; tt={1,3,5} → patched rate {0.50, 0.50, 0.50}; restored 0, lost 0
- L24, source proj +0.948; tt={1,3,5} → patched rate {0.40, 0.40, 0.50}; restored 0, lost 2
- L27, source proj −0.458; tt={1,3,5} → patched rate {0.40, 0.50, 0.50}; restored 0, lost 1
- Totals across 90 records: 0 restored, 3 lost, 87 no change.

## Compile command

From `report6/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report6.tex
bibtex report6 || true
pdflatex -interaction=nonstopmode -halt-on-error report6.tex
pdflatex -interaction=nonstopmode -halt-on-error report6.tex
```

This produced `report6.pdf` cleanly. No undefined references, no undefined citations.

## Warnings (non-blocking)

- A handful of `Underfull \hbox` and `Overfull \hbox` paragraph-fitting warnings — purely cosmetic and produce no visible problem in the PDF.
- Several `'h' float specifier changed to 'ht'` notices — LaTeX promoting `[h]` placement to `[ht]`, which is fine for this layout.
- Initial compile required adding `\usepackage{amsmath}` to support `\text{...}` in subscripts; revision pass also added `\usepackage{placeins}` for `\FloatBarrier`. No other style/package changes were needed.

## Page count and structure

7 pages total, well within the 4–8 page rubric. Section order:
1. Revised Project Framing (system, task, environment, phenomenon, RQ, hypotheses).
2. High-Level Research Design.
3. Behavioral Results.
4. Refusal-Direction Tracing Results.
5. Patching Results.
6. Updated Hypothesis Status.
7. Conclusions and Threats to Validity (internal / external / construct).
+ References.

## What was missing or uncertain

- **Behavioral k=10 inside the R6 pipeline.** The R6 generation cached error records on a GTX 1080 Ti; the paper cites the R3 pilot number (9/25=0.36) and labels it as exploratory reference data only. Re-running with `--no-resume` on an L40S would close this gap; it is logged in the Report 7 next-steps list.
- **VERIFY_REPORT6.md is partly stale.** It claims all four R6 generations produced only error records. That is true only for harmful_k10. The CSV `generation_refusal_rates_by_condition.csv` and the bar chart at `outputs/report6/plots/report6_refusal_rate_by_condition.png` (May 6 19:38 timestamp) are valid for the three completed conditions. The report uses the more recent `REPORT3_REPORT6_ANALYSIS.md` as the authoritative source; if this is later contradicted by direct re-inspection, the behavioral numbers should be re-checked.
- **Direction extraction is not held out** (the 25 traced prompts are a subset of the 50 prompts used to extract the direction). Listed as a Section 7 internal-validity threat.
- **Prompt-level association inside k=3** is not run; listed as a Report 7 priority. The data to do this exists in `outputs/report6/summaries/trace_prompt_level_key_layers.csv`.
- **Phrase-list classifier** is the only refusal label source; no LlamaGuard cross-check.

## Optional cleanup

The `outputs/report6/plots/` files used in the report are mirrored into `report6/figures/` so the build is self-contained. If you would prefer the LaTeX to point at the canonical pipeline outputs instead, change the `\includegraphics` paths from `figures/...` to `../outputs/report6/plots/...` and rebuild.
