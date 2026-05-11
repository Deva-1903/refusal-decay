#!/usr/bin/env python3
"""
Prompt-level association between attacked behavior labels and late-layer projection.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Report 7 prompt-level projection/label association.")
    parser.add_argument("--summary-dir", type=str, default="outputs/report7/summaries")
    parser.add_argument("--fallback-report6-summary-dir", type=str, default="outputs/report6/summaries")
    parser.add_argument("--condition", type=str, default="harmful_k03")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 27])
    return parser.parse_args()


def _read_with_fallback(filename: str, primary: Path, fallback: Path) -> pd.DataFrame:
    primary_path = primary / filename
    if primary_path.exists():
        return pd.read_csv(primary_path)
    fallback_path = fallback / filename
    if fallback_path.exists():
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(f"Missing {filename} in {primary} and {fallback}")


def main() -> None:
    args = parse_args()
    setup_logging(log_file=Path(args.summary_dir) / "report7_prompt_association.log")

    summary_dir = ensure_dir(args.summary_dir)
    fallback_dir = Path(args.fallback_report6_summary_dir)
    labels = _read_with_fallback("generation_prompt_labels.csv", summary_dir, fallback_dir)
    prompt_projection = _read_with_fallback("trace_prompt_level_key_layers.csv", summary_dir, fallback_dir)

    labels = labels[
        (labels["condition_name"] == args.condition)
        & (labels["refusal_phrase_label"].isin(["refusal", "compliance"]))
    ][["prompt_id", "condition_name", "refusal_phrase_label"]].drop_duplicates()

    merged = prompt_projection.merge(labels, on="prompt_id", how="inner")
    # Column-name resolution after merge:
    # - If pandas added suffixes (both sides had `condition_name`), prefer the
    #   x-suffixed copy (prompt_projection side).
    # - If `condition` already exists from the prompt-level summary, drop the
    #   labels-side `condition_name` rather than renaming into a duplicate.
    # - Otherwise rename `condition_name` to `condition` as before.
    if "condition_name_x" in merged.columns:
        merged = merged.rename(columns={"condition_name_x": "condition"})
        if "condition_name_y" in merged.columns:
            merged = merged.drop(columns=["condition_name_y"])
    elif "condition" in merged.columns and "condition_name" in merged.columns:
        merged = merged.drop(columns=["condition_name"])
    elif "condition_name" in merged.columns:
        merged = merged.rename(columns={"condition_name": "condition"})
    merged = merged[(merged["condition"] == args.condition) & (merged["layer"].isin(args.layers))]

    assoc = (
        merged.groupby(["condition", "layer", "refusal_phrase_label"])
        .agg(
            n_prompts=("prompt_id", "nunique"),
            mean_projection=("mean_generated_projection", "mean"),
            median_projection=("mean_generated_projection", "median"),
            std_projection=("mean_generated_projection", "std"),
            min_projection=("mean_generated_projection", "min"),
            max_projection=("mean_generated_projection", "max"),
        )
        .reset_index()
        .rename(columns={"refusal_phrase_label": "label_group"})
    )

    rows: list[dict] = []
    for (condition, layer), group in assoc.groupby(["condition", "layer"]):
        lookup = {row.label_group: row for row in group.itertuples()}
        if "compliance" not in lookup or "refusal" not in lookup:
            continue
        rows.append(
            {
                "condition": condition,
                "layer": layer,
                "compliance_n": int(lookup["compliance"].n_prompts),
                "refusal_n": int(lookup["refusal"].n_prompts),
                "compliance_mean_projection": float(lookup["compliance"].mean_projection),
                "refusal_mean_projection": float(lookup["refusal"].mean_projection),
                "compliance_minus_refusal_mean": float(
                    lookup["compliance"].mean_projection - lookup["refusal"].mean_projection
                ),
            }
        )
    differences = pd.DataFrame(rows)

    assoc_path = summary_dir / "prompt_projection_label_association.csv"
    diff_path = summary_dir / "prompt_projection_label_differences.csv"
    assoc.to_csv(assoc_path, index=False)
    differences.to_csv(diff_path, index=False)
    print(assoc_path)
    print(diff_path)


if __name__ == "__main__":
    main()
