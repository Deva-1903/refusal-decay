#!/usr/bin/env python3
"""
Build Report 6 summary tables and a small manifest of saved artifacts.
"""

import argparse
from collections import OrderedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config
from src.utils.io_utils import ensure_dir, load_jsonl, save_json
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Report 6 outputs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/report6/generation_report6.yaml",
        help="Report 6 master config.",
    )
    return parser.parse_args()


def _condition_order(cfg) -> list[str]:
    conditions = list(getattr(cfg.report6, "generation_conditions", []))
    return [condition["name"] for condition in conditions]


def _condition_sort_key(order: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(order)}


def _format_condition_name(dataset_name: str | None, prefix_k) -> str | None:
    if dataset_name is None or prefix_k is None:
        return None
    return f"{dataset_name}_k{int(prefix_k):02d}"


def _load_generation_records(generations_dir: Path) -> list[dict]:
    combined_path = generations_dir / "report6_generation_combined.jsonl"
    if combined_path.exists():
        return load_jsonl(combined_path)

    records: list[dict] = []
    for classified_path in sorted(generations_dir.glob("*/classified.jsonl")):
        records.extend(load_jsonl(classified_path))
    return records


def _load_trace_df(trace_dir: Path) -> pd.DataFrame | None:
    parquet_path = trace_dir / "traces_all.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = trace_dir / "traces_all.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return None


def _load_patch_records(patch_dir: Path) -> list[dict]:
    classified_path = patch_dir / "patching_classified.jsonl"
    if not classified_path.exists():
        return []
    return load_jsonl(classified_path)


def _attach_condition_columns(df: pd.DataFrame, dataset_fallback_col: str = "label") -> pd.DataFrame:
    if df.empty:
        return df

    if "dataset_name" not in df.columns and dataset_fallback_col in df.columns:
        df["dataset_name"] = df[dataset_fallback_col]

    if "condition_name" not in df.columns:
        df["condition_name"] = df.apply(
            lambda row: _format_condition_name(
                row.get("dataset_name"),
                row.get("prefix_k"),
            ),
            axis=1,
        )

    return df


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def summarize_generation(records: list[dict], summary_dir: Path, sort_lookup: dict[str, int]) -> list[Path]:
    if not records:
        raise FileNotFoundError(
            "No Report 6 generation outputs found. Run scripts/run_report6_generation.py first."
        )

    df = pd.DataFrame(records)
    df = _attach_condition_columns(df)
    df["prefix_k"] = df["prefix_k"].astype(int)

    summary = (
        df.groupby(["condition_name", "dataset_name", "prefix_k"])
        .agg(
            n_total=("prompt_id", "size"),
            n_refusal=("refusal_phrase_label", lambda x: (x == "refusal").sum()),
            n_compliance=("refusal_phrase_label", lambda x: (x == "compliance").sum()),
            n_error=("refusal_phrase_label", lambda x: (x == "error").sum()),
        )
        .reset_index()
    )
    summary["n_valid"] = summary["n_refusal"] + summary["n_compliance"]
    summary["refusal_rate"] = summary["n_refusal"] / summary["n_valid"].where(summary["n_valid"] > 0)
    summary["condition_sort"] = summary["condition_name"].map(lambda value: sort_lookup.get(value, 999))
    summary = summary.sort_values(["condition_sort", "prefix_k", "dataset_name"]).drop(columns=["condition_sort"])

    labels = (
        df[["prompt_id", "condition_name", "dataset_name", "prefix_k", "refusal_phrase_label", "matched_phrase"]]
        .sort_values(["condition_name", "prompt_id"])
        .reset_index(drop=True)
    )

    summary_path = summary_dir / "generation_refusal_rates_by_condition.csv"
    labels_path = summary_dir / "generation_prompt_labels.csv"
    _save_df(summary, summary_path)
    _save_df(labels, labels_path)
    return [summary_path, labels_path]


def summarize_traces(
    trace_df: pd.DataFrame | None,
    summary_dir: Path,
    key_layers: list[int],
    sort_lookup: dict[str, int],
) -> list[Path]:
    if trace_df is None or trace_df.empty:
        return []

    df = trace_df.copy()
    df = _attach_condition_columns(df)
    df["prefix_k"] = df["prefix_k"].astype(int)
    df["layer"] = df["layer"].astype(int)

    generated_df = df[df["is_prefill"] == False].copy()  # noqa: E712
    if generated_df.empty:
        return []

    token_summary = (
        generated_df.groupby(["condition_name", "prefix_k", "layer", "gen_token_pos"])
        .agg(
            mean_projection=("projection", "mean"),
            std_projection=("projection", "std"),
            n_records=("projection", "size"),
            n_prompts=("prompt_id", "nunique"),
        )
        .reset_index()
    )
    token_summary["condition_sort"] = token_summary["condition_name"].map(
        lambda value: sort_lookup.get(value, 999)
    )
    token_summary = token_summary.sort_values(
        ["condition_sort", "layer", "gen_token_pos"]
    ).drop(columns=["condition_sort"])

    layer_summary = (
        generated_df.groupby(["condition_name", "prefix_k", "layer"])
        .agg(
            mean_projection=("projection", "mean"),
            std_projection=("projection", "std"),
            n_records=("projection", "size"),
            n_prompts=("prompt_id", "nunique"),
        )
        .reset_index()
    )
    layer_summary["condition_sort"] = layer_summary["condition_name"].map(
        lambda value: sort_lookup.get(value, 999)
    )
    layer_summary = layer_summary.sort_values(["condition_sort", "layer"]).drop(columns=["condition_sort"])

    key_df = df[df["layer"].isin(key_layers)].copy()
    prompt_stats = (
        key_df[key_df["is_prefill"] == False]  # noqa: E712
        .groupby(["condition_name", "prompt_id", "layer"])
        .agg(
            mean_generated_projection=("projection", "mean"),
            min_generated_projection=("projection", "min"),
            max_generated_projection=("projection", "max"),
            n_generated_positions=("projection", "size"),
        )
        .reset_index()
    )

    final_generated = (
        key_df[key_df["is_prefill"] == False]  # noqa: E712
        .sort_values(["condition_name", "prompt_id", "layer", "gen_token_pos"])
        .groupby(["condition_name", "prompt_id", "layer"])
        .tail(1)[["condition_name", "prompt_id", "layer", "gen_token_pos", "projection"]]
        .rename(
            columns={
                "gen_token_pos": "final_generated_token_pos",
                "projection": "final_generated_projection",
            }
        )
    )

    prefill_projection = (
        key_df[key_df["is_prefill"] == True]  # noqa: E712
        .sort_values(["condition_name", "prompt_id", "layer", "step"])
        .groupby(["condition_name", "prompt_id", "layer"])
        .tail(1)[["condition_name", "prompt_id", "layer", "projection"]]
        .rename(columns={"projection": "prefill_projection"})
    )

    prompt_summary = prompt_stats.merge(
        final_generated,
        on=["condition_name", "prompt_id", "layer"],
        how="left",
    ).merge(
        prefill_projection,
        on=["condition_name", "prompt_id", "layer"],
        how="left",
    )
    prompt_summary["condition_sort"] = prompt_summary["condition_name"].map(
        lambda value: sort_lookup.get(value, 999)
    )
    prompt_summary = prompt_summary.sort_values(
        ["condition_sort", "layer", "prompt_id"]
    ).drop(columns=["condition_sort"])

    token_path = summary_dir / "trace_projection_by_condition_layer_token.csv"
    layer_path = summary_dir / "trace_mean_projection_by_condition_layer.csv"
    prompt_path = summary_dir / "trace_prompt_level_key_layers.csv"
    _save_df(token_summary, token_path)
    _save_df(layer_summary, layer_path)
    _save_df(prompt_summary, prompt_path)
    return [token_path, layer_path, prompt_path]


def summarize_patching(
    patch_records: list[dict],
    summary_dir: Path,
    sort_lookup: dict[str, int],
) -> list[Path]:
    if not patch_records:
        return []

    df = pd.DataFrame(patch_records)
    if df.empty:
        return []

    df = _attach_condition_columns(df, dataset_fallback_col="dataset_name")
    valid_df = df[df.get("error").isna()] if "error" in df.columns else df
    if valid_df.empty:
        return []

    summary = (
        valid_df.groupby(["condition_name", "layer", "source_position", "target_position", "mode"])
        .agg(
            n_total=("prompt_id", "size"),
            patch_applied_rate=("patch_applied", "mean"),
            mean_source_projection=("source_projection", "mean"),
            baseline_refusal_rate=("baseline_phrase_label", lambda x: (x == "refusal").mean()),
            patched_refusal_rate=("patched_phrase_label", lambda x: (x == "refusal").mean()),
            n_refusal_restored=("refusal_restored", "sum"),
            n_refusal_lost=("refusal_lost", "sum"),
        )
        .reset_index()
    )
    summary["n_no_change"] = (
        summary["n_total"] - summary["n_refusal_restored"] - summary["n_refusal_lost"]
    )
    summary["refusal_rate_delta"] = (
        summary["patched_refusal_rate"] - summary["baseline_refusal_rate"]
    )
    summary["condition_sort"] = summary["condition_name"].map(lambda value: sort_lookup.get(value, 999))
    summary = summary.sort_values(
        ["condition_sort", "layer", "target_position"]
    ).drop(columns=["condition_sort"])

    prompt_path = summary_dir / "patching_prompt_results.csv"
    summary_path = summary_dir / "patching_refusal_recovery_summary.csv"
    _save_df(valid_df.sort_values(["layer", "target_position", "prompt_id"]), prompt_path)
    _save_df(summary, summary_path)
    return [summary_path, prompt_path]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    log_file = Path(cfg.output.log_dir) / "report6_summary.log"
    setup_logging(level=getattr(cfg.logging, "level", "INFO"), log_file=log_file)

    import logging

    logger = logging.getLogger(__name__)

    generations_dir = Path(cfg.output.generations_dir)
    traces_dir = Path(cfg.tracing.output_dir)
    patching_dir = Path(cfg.patching.output_dir)
    summary_dir = ensure_dir(getattr(cfg.report6, "summaries_dir", "outputs/report6/summaries"))

    order = _condition_order(cfg)
    sort_lookup = _condition_sort_key(order)

    logger.info("Building Report 6 summaries.")
    logger.info("Generations dir: %s", generations_dir)
    logger.info("Traces dir: %s", traces_dir)
    logger.info("Patching dir: %s", patching_dir)
    logger.info("Summary dir: %s", summary_dir)

    created_paths: list[Path] = []
    created_paths.extend(summarize_generation(_load_generation_records(generations_dir), summary_dir, sort_lookup))
    created_paths.extend(
        summarize_traces(
            _load_trace_df(traces_dir),
            summary_dir,
            key_layers=list(getattr(cfg.report6, "key_layers", [24, 27])),
            sort_lookup=sort_lookup,
        )
    )
    created_paths.extend(
        summarize_patching(
            _load_patch_records(patching_dir),
            summary_dir,
            sort_lookup=sort_lookup,
        )
    )

    manifest = OrderedDict(
        config=args.config,
        summary_dir=str(summary_dir),
        created_files=[str(path) for path in created_paths],
    )
    manifest_path = summary_dir / "report6_manifest.json"
    save_json(manifest, manifest_path)
    created_paths.append(manifest_path)

    logger.info("Created %d summary artifacts.", len(created_paths))
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
