#!/usr/bin/env python3
"""
Compare Report 6 and Report 7 intervention results in one table.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Report 6 and Report 7 interventions.")
    parser.add_argument("--report6-summary", type=str, default="outputs/report6/summaries/patching_refusal_recovery_summary.csv")
    parser.add_argument("--report7-cross-summary", type=str, default="outputs/report7/patching/cross_condition_patching_summary.csv")
    parser.add_argument("--report7-additive-summary", type=str, default="outputs/report7/interventions/additive_direction_summary.csv")
    parser.add_argument("--output", type=str, default="outputs/report7/summaries/report6_vs_report7_intervention_comparison.csv")
    return parser.parse_args()


def _append_report6(rows: list[dict], path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        n = int(row.n_total)
        restored = int(row.n_refusal_restored)
        lost = int(row.n_refusal_lost)
        unchanged = int(row.n_no_change)
        rows.append(
            {
                "method": "report6_same_condition_attacked_source",
                "layer": int(row.layer),
                "target_position": int(row.target_position),
                "n_prompts": n,
                "restored": restored,
                "lost": lost,
                "unchanged": unchanged,
                "restoration_rate": restored / n if n else None,
                "loss_rate": lost / n if n else None,
            }
        )


def _append_report7_cross(rows: list[dict], path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        n = int(row.n_prompts)
        rows.append(
            {
                "method": "report7_baseline_source_cross_condition",
                "layer": int(row.layer),
                "target_position": int(row.target_position),
                "n_prompts": n,
                "restored": int(row.restored),
                "lost": int(row.lost),
                "unchanged": int(row.unchanged),
                "restoration_rate": float(row.restoration_rate),
                "loss_rate": float(row.loss_rate),
            }
        )


def _append_report7_additive(rows: list[dict], path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "direction_type" in df.columns:
        df = df[df["direction_type"] == "refusal"]
    for row in df.itertuples(index=False):
        n = int(row.n_prompts)
        rows.append(
            {
                "method": f"report7_additive_refusal_alpha_{float(row.alpha):g}",
                "layer": int(row.layer),
                "target_position": int(row.target_position),
                "n_prompts": n,
                "restored": int(row.restored),
                "lost": int(row.lost),
                "unchanged": int(row.unchanged),
                "restoration_rate": float(row.restoration_rate),
                "loss_rate": float(row.loss_rate),
            }
        )


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    _append_report6(rows, Path(args.report6_summary))
    _append_report7_cross(rows, Path(args.report7_cross_summary))
    _append_report7_additive(rows, Path(args.report7_additive_summary))

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
