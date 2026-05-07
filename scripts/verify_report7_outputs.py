#!/usr/bin/env python3
"""
Verify Report 7 outputs and write VERIFY_REPORT7.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Report 7 outputs.")
    parser.add_argument("--output-root", type=str, default="outputs/report7")
    parser.add_argument("--report-path", type=str, default="VERIFY_REPORT7.md")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check_generation(root: Path) -> tuple[list[str], list[str], list[str]]:
    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    for condition in ["harmful_k00", "harmful_k03", "benign_k00"]:
        path = root / "generations" / condition / "classified.jsonl"
        rows = _load_jsonl(path)
        if not rows:
            missing.append(f"Missing generation cache: {path}")
            continue
        errors = sum(1 for row in rows if row.get("error") or row.get("refusal_phrase_label") == "error")
        if errors == len(rows):
            suspicious.append(f"All-error generation cache: {path}")
        else:
            complete.append(f"{condition}: {len(rows)} rows, {errors} errors")
    return complete, missing, suspicious


def _check_tracing(root: Path) -> tuple[list[str], list[str], list[str]]:
    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    path = root / "summaries" / "trace_generated_token_mean_by_condition_layer.csv"
    if not path.exists():
        missing.append(f"Missing generated-token tracing summary: {path}")
        return complete, missing, suspicious
    df = pd.read_csv(path)
    layers = set(df["layer"].astype(int))
    conditions = set(df["condition"])
    expected_layers = {16, 20, 24, 27}
    expected_conditions = {"harmful_k00", "harmful_k03"}
    if expected_layers.issubset(layers) and expected_conditions.issubset(conditions):
        complete.append("Tracing summary covers layers 16/20/24/27 and harmful_k00/harmful_k03.")
    else:
        suspicious.append(f"Tracing coverage incomplete: layers={sorted(layers)}, conditions={sorted(conditions)}")
    return complete, missing, suspicious


def _check_prompt_association(root: Path) -> tuple[list[str], list[str], list[str]]:
    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    path = root / "summaries" / "prompt_projection_label_association.csv"
    if not path.exists():
        missing.append(f"Missing prompt association CSV: {path}")
        return complete, missing, suspicious
    df = pd.read_csv(path)
    layers = set(df["layer"].astype(int)) if not df.empty else set()
    groups = set(df["label_group"]) if not df.empty else set()
    if {24, 27}.issubset(layers) and {"refusal", "compliance"}.issubset(groups):
        complete.append("Prompt association includes layers 24/27 and both label groups.")
    else:
        suspicious.append(f"Prompt association may be thin: layers={sorted(layers)}, groups={sorted(groups)}")
    return complete, missing, suspicious


def _check_cross_condition(root: Path) -> tuple[list[str], list[str], list[str]]:
    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    path = root / "patching" / "cross_condition_patching_results.csv"
    if not path.exists():
        missing.append(f"Missing cross-condition patching results: {path}")
        return complete, missing, suspicious
    df = pd.read_csv(path)
    valid = df[df["error"].fillna("") == ""]
    layers = set(valid["layer"].astype(int)) if not valid.empty else set()
    targets = set(valid["target_position"].astype(int)) if not valid.empty else set()
    if {16, 24, 27}.issubset(layers) and targets:
        complete.append(f"Cross-condition patching valid rows={len(valid)}, layers={sorted(layers)}, targets={sorted(targets)}")
    else:
        suspicious.append(f"Cross-condition coverage incomplete: valid={len(valid)}, layers={sorted(layers)}, targets={sorted(targets)}")
    if valid.empty or not {"restored_refusal", "lost_refusal", "unchanged"}.issubset(valid.columns):
        suspicious.append("Cross-condition restored/lost/unchanged columns are missing or not computable.")
    return complete, missing, suspicious


def _check_additive(root: Path) -> tuple[list[str], list[str], list[str]]:
    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    path = root / "interventions" / "additive_direction_results.csv"
    if not path.exists():
        missing.append(f"Missing additive intervention results: {path}")
        return complete, missing, suspicious
    df = pd.read_csv(path)
    valid = df[df["error"].fillna("") == ""]
    alphas = set(valid["alpha"].astype(float)) if not valid.empty else set()
    direction_types = set(valid["direction_type"]) if not valid.empty else set()
    if {0.5, 1.0, 2.0}.issubset(alphas) and "refusal" in direction_types:
        complete.append(f"Additive intervention valid rows={len(valid)}, alphas={sorted(alphas)}, directions={sorted(direction_types)}")
    else:
        suspicious.append(f"Additive coverage incomplete: alphas={sorted(alphas)}, directions={sorted(direction_types)}")
    return complete, missing, suspicious


def main() -> int:
    args = parse_args()
    root = ROOT / args.output_root
    checks = [
        ("Behavioral", _check_generation(root)),
        ("Tracing", _check_tracing(root)),
        ("Prompt-Level Association", _check_prompt_association(root)),
        ("Cross-Condition Patching", _check_cross_condition(root)),
        ("Additive Intervention", _check_additive(root)),
    ]

    complete: list[str] = []
    missing: list[str] = []
    suspicious: list[str] = []
    lines = ["# Verify Report 7", ""]
    for title, (done, miss, sus) in checks:
        complete.extend([f"{title}: {item}" for item in done])
        missing.extend([f"{title}: {item}" for item in miss])
        suspicious.extend([f"{title}: {item}" for item in sus])

    ready = not missing and not suspicious
    lines.extend(["## COMPLETE", ""])
    lines.extend([f"- {item}" for item in complete] or ["- None yet"])
    lines.extend(["", "## MISSING", ""])
    lines.extend([f"- {item}" for item in missing] or ["- None"])
    lines.extend(["", "## SUSPICIOUS", ""])
    lines.extend([f"- {item}" for item in suspicious] or ["- None"])
    lines.extend(["", "## READY TO WRITE REPORT 7?", ""])
    lines.append("Yes" if ready else "Not yet")
    lines.extend(["", "## RUN THESE NEXT", ""])
    if missing or suspicious:
        incomplete = missing + suspicious
        next_steps: list[str] = []
        if any(item.startswith("Behavioral:") for item in incomplete):
            next_steps.append(
                "`python scripts/run_report7_generation.py --config configs/experiments/report7/generation_report7.yaml --conditions harmful_k00 harmful_k03 benign_k00`"
            )
        if any(item.startswith("Tracing:") for item in incomplete):
            next_steps.append(
                "`python scripts/summarize_report7_tracing.py --config configs/experiments/report7/tracing_report7.yaml`"
            )
        if any(item.startswith("Prompt-Level Association:") for item in incomplete):
            next_steps.append("`python scripts/analyze_report7_prompt_association.py`")
        if any(item.startswith("Cross-Condition Patching:") for item in incomplete):
            next_steps.append(
                "`python scripts/run_report7_cross_condition_patching.py --config configs/experiments/report7/patching_report7.yaml --direction-path outputs/report7/directions/refusal_direction.pt`"
            )
        if any(item.startswith("Additive Intervention:") for item in incomplete):
            next_steps.append(
                "`python scripts/run_report7_additive_direction_intervention.py --config configs/experiments/report7/additive_intervention_report7.yaml --direction-path outputs/report7/directions/refusal_direction.pt`"
            )
        next_steps.extend(
            [
                "`python scripts/compare_report6_report7_patching.py`",
                "`python scripts/plot_report7_results.py`",
                "`python scripts/verify_report7_outputs.py`",
            ]
        )
        lines.extend([f"- {step}" for step in dict.fromkeys(next_steps)])
    else:
        lines.append("- Report 7 outputs are ready for writeup.")

    report_path = ROOT / args.report_path
    report_path.write_text("\n".join(lines) + "\n")

    print("COMPLETE")
    for item in complete:
        print(f"- {item}")
    print("MISSING")
    for item in missing:
        print(f"- {item}")
    print("SUSPICIOUS")
    for item in suspicious:
        print(f"- {item}")
    print(f"READY TO WRITE REPORT 7? {'YES' if ready else 'NO'}")
    print("RUN THESE NEXT")
    for line in lines[lines.index("## RUN THESE NEXT") + 2 :]:
        if line.startswith("- "):
            print(line)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
