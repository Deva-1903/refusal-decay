#!/usr/bin/env python3
"""
Build standardized Report 7 tracing summaries.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Report 7 tracing outputs.")
    parser.add_argument("--config", type=str, default="configs/experiments/report7/tracing_report7.yaml")
    return parser.parse_args()


def _load_trace_df(cfg) -> pd.DataFrame:
    candidates = [
        Path(cfg.tracing.output_dir) / "traces_all.parquet",
        Path(cfg.tracing.output_dir) / "traces_all.csv",
        Path(getattr(cfg.report7, "source_report6_trace_dir", "outputs/report6/traces")) / "traces_all.parquet",
        Path(getattr(cfg.report7, "source_report6_trace_dir", "outputs/report6/traces")) / "traces_all.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
    raise FileNotFoundError("No traces_all parquet/csv found for Report 7 or Report 6 fallback.")


def _attach_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition_name" not in df.columns:
        df["condition_name"] = df.apply(
            lambda row: f"{row.get('label', 'harmful')}_k{int(row['prefix_k']):02d}",
            axis=1,
        )
    return df.rename(columns={"condition_name": "condition"})


def _aggregate(df: pd.DataFrame, group_cols: list[str], token_window: str) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)
        .agg(
            n_prompts=("prompt_id", "nunique"),
            n_rows=("projection", "size"),
            mean_projection=("projection", "mean"),
            std_projection=("projection", "std"),
            min_projection=("projection", "min"),
            max_projection=("projection", "max"),
        )
        .reset_index()
    )
    out["token_window"] = token_window
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=getattr(cfg.logging, "level", "INFO"), log_file=Path(cfg.output.log_dir) / "report7_tracing_summary.log")

    # Honor a config-provided summaries dir (paper runs write to outputs/paper);
    # default to the Report 7 location for backward compatibility.
    summary_dir = ensure_dir(getattr(getattr(cfg, "report7", object()), "summaries_dir", "outputs/report7/summaries"))
    df = _attach_condition(_load_trace_df(cfg))
    df["layer"] = df["layer"].astype(int)

    generated = df[df["is_prefill"] == False].copy()  # noqa: E712
    generated_mean = _aggregate(generated, ["condition", "layer"], "generated_tokens")
    all_position_mean = _aggregate(df, ["condition", "layer"], "all_positions")

    trajectory = _aggregate(generated, ["condition", "layer", "gen_token_pos"], "generated_token")
    trajectory = trajectory.rename(columns={"gen_token_pos": "token_position"})

    # Prompt-level summary required by analyze_report7_prompt_association.py.
    # One row per (condition, layer, prompt_id) with the mean of generated-token
    # projections — this is what H3 (refusers vs. compliers) is computed on.
    prompt_level = (
        generated.groupby(["condition", "layer", "prompt_id"])
        .agg(
            n_rows=("projection", "size"),
            mean_generated_projection=("projection", "mean"),
            std_generated_projection=("projection", "std"),
            min_generated_projection=("projection", "min"),
            max_generated_projection=("projection", "max"),
        )
        .reset_index()
    )

    generated_path = summary_dir / "trace_generated_token_mean_by_condition_layer.csv"
    all_path = summary_dir / "trace_all_position_mean_by_condition_layer.csv"
    trajectory_path = summary_dir / "trace_token_trajectory_by_condition_layer.csv"
    prompt_level_path = summary_dir / "trace_prompt_level_key_layers.csv"
    generated_mean.to_csv(generated_path, index=False)
    all_position_mean.to_csv(all_path, index=False)
    trajectory.to_csv(trajectory_path, index=False)
    prompt_level.to_csv(prompt_level_path, index=False)

    for path in [generated_path, all_path, trajectory_path, prompt_level_path]:
        print(path)


if __name__ == "__main__":
    main()
