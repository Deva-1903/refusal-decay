#!/usr/bin/env python3
"""
Generate the minimum report-ready figures for Report 6.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Report 6 results.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/report6/generation_report6.yaml",
        help="Report 6 master config.",
    )
    return parser.parse_args()


def _condition_order(cfg) -> list[str]:
    return [condition["name"] for condition in getattr(cfg.report6, "generation_conditions", [])]


def _load_summary_csv(summary_dir: Path, filename: str) -> pd.DataFrame | None:
    path = summary_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_trace_df(trace_dir: Path) -> pd.DataFrame | None:
    parquet_path = trace_dir / "traces_all.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = trace_dir / "traces_all.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)

    if "condition_name" not in df.columns and {"label", "prefix_k"}.issubset(df.columns):
        df["condition_name"] = df.apply(
            lambda row: f"{row['label']}_k{int(row['prefix_k']):02d}",
            axis=1,
        )
    return df


def _condition_label(name: str) -> str:
    match = re.match(r"([a-zA-Z]+)_k(\d+)", name)
    if not match:
        return name
    dataset, k = match.groups()
    return f"{dataset} k={int(k)}"


def _savefig(fig: plt.Figure, path: Path, saved_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)


def plot_refusal_rate(summary_df: pd.DataFrame, order: list[str], plot_dir: Path, saved_paths: list[Path]) -> None:
    if summary_df is None or summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["condition_sort"] = plot_df["condition_name"].map(
        lambda value: order.index(value) if value in order else 999
    )
    plot_df = plot_df.sort_values("condition_sort")
    colors = ["#b91c1c" if dataset == "harmful" else "#2563eb" for dataset in plot_df["dataset_name"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        np.arange(len(plot_df)),
        plot_df["refusal_rate"] * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels([_condition_label(name) for name in plot_df["condition_name"]], rotation=20, ha="right")
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Report 6: Refusal Rate by Condition")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, plot_dir / "report6_refusal_rate_by_condition.png", saved_paths)


def plot_heatmap_comparison(
    token_df: pd.DataFrame,
    cfg,
    plot_dir: Path,
    saved_paths: list[Path],
) -> None:
    if token_df is None or token_df.empty:
        return

    attacked_name = f"harmful_k{int(getattr(cfg.report6, 'attacked_k', 3)):02d}"
    optional_name = f"harmful_k{int(getattr(cfg.report6, 'optional_k', 10)):02d}"
    condition_names = ["harmful_k00", attacked_name]
    if optional_name in set(token_df["condition_name"]):
        condition_names.append(optional_name)

    max_pos = int(getattr(cfg.report6, "plot_max_token_positions", 32))
    plot_df = token_df[token_df["gen_token_pos"] < max_pos].copy()
    if plot_df.empty:
        return

    selected = plot_df[plot_df["condition_name"].isin(condition_names)].copy()
    if selected.empty:
        return

    vmax = float(np.nanmax(np.abs(selected["mean_projection"].to_numpy())))
    fig, axes = plt.subplots(1, len(condition_names), figsize=(5.2 * len(condition_names), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, condition_name in zip(axes, condition_names):
        sub = selected[selected["condition_name"] == condition_name]
        pivot = (
            sub.pivot_table(
                index="layer",
                columns="gen_token_pos",
                values="mean_projection",
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(_condition_label(condition_name))
        ax.set_xlabel("Generated token position")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Layer")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Mean projection")
    fig.suptitle("Report 6: Layer x Token Projection Heatmaps", y=1.02)
    _savefig(fig, plot_dir / "report6_heatmap_baseline_vs_prefill.png", saved_paths)


def plot_layer_line(
    token_df: pd.DataFrame,
    layer: int,
    title: str,
    filename: str,
    cfg,
    plot_dir: Path,
    saved_paths: list[Path],
) -> None:
    if token_df is None or token_df.empty:
        return

    plot_df = token_df[token_df["layer"] == layer].copy()
    if plot_df.empty:
        return

    order = [name for name in _condition_order(cfg) if name in set(plot_df["condition_name"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#111827", "#b91c1c", "#f59e0b", "#2563eb"]

    for idx, condition_name in enumerate(order):
        sub = plot_df[plot_df["condition_name"] == condition_name].sort_values("gen_token_pos")
        ax.plot(
            sub["gen_token_pos"],
            sub["mean_projection"],
            label=_condition_label(condition_name),
            linewidth=2,
            color=colors[idx % len(colors)],
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Mean refusal-direction projection")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    _savefig(fig, plot_dir / filename, saved_paths)


def plot_prompt_level_raw(
    prompt_summary_df: pd.DataFrame | None,
    raw_trace_df: pd.DataFrame | None,
    cfg,
    plot_dir: Path,
    saved_paths: list[Path],
) -> None:
    if prompt_summary_df is None or prompt_summary_df.empty or raw_trace_df is None or raw_trace_df.empty:
        return

    plot_layer = int(getattr(cfg.report6, "prompt_plot_layer", 27))
    attacked_name = f"harmful_k{int(getattr(cfg.report6, 'attacked_k', 3)):02d}"
    prompt_df = prompt_summary_df[
        (prompt_summary_df["layer"] == plot_layer)
        & (prompt_summary_df["condition_name"].isin(["harmful_k00", attacked_name]))
    ].copy()
    if prompt_df.empty:
        return

    pivot = prompt_df.pivot_table(
        index="prompt_id",
        columns="condition_name",
        values="mean_generated_projection",
        aggfunc="first",
    )
    if "harmful_k00" not in pivot.columns or attacked_name not in pivot.columns:
        return

    pivot["delta"] = pivot[attacked_name] - pivot["harmful_k00"]
    prompt_id = pivot["delta"].idxmin()

    raw_df = raw_trace_df[
        (raw_trace_df["prompt_id"] == prompt_id)
        & (raw_trace_df["layer"] == plot_layer)
        & (raw_trace_df["condition_name"].isin(["harmful_k00", attacked_name]))
    ].copy()
    if raw_df.empty:
        return

    raw_df["plot_token_pos"] = raw_df["gen_token_pos"]
    raw_df.loc[raw_df["is_prefill"] == True, "plot_token_pos"] = -1  # noqa: E712

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition_name, color in [("harmful_k00", "#111827"), (attacked_name, "#b91c1c")]:
        sub = raw_df[raw_df["condition_name"] == condition_name].sort_values("plot_token_pos")
        ax.plot(
            sub["plot_token_pos"],
            sub["projection"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=_condition_label(condition_name),
            color=color,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Token position (-1 = last input/prefill token)")
    ax.set_ylabel("Projection")
    ax.set_title(f"Report 6: Prompt-Level Layer {plot_layer} Trace ({prompt_id})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    filename = f"report6_prompt_level_layer{plot_layer}_{prompt_id}.png"
    _savefig(fig, plot_dir / filename, saved_paths)


def plot_patching(summary_df: pd.DataFrame | None, plot_dir: Path, saved_paths: list[Path]) -> None:
    if summary_df is None or summary_df.empty:
        return

    plot_df = summary_df.sort_values(["layer", "target_position"]).copy()
    labels = [
        f"L{int(row.layer)} / t{int(row.target_position)}"
        for _, row in plot_df.iterrows()
    ]
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        x - width / 2,
        plot_df["baseline_refusal_rate"] * 100,
        width,
        label="Attacked no patch",
        color="#9ca3af",
    )
    ax.bar(
        x + width / 2,
        plot_df["patched_refusal_rate"] * 100,
        width,
        label="Patched",
        color="#2563eb",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Report 6: Targeted Patching Comparison")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    _savefig(fig, plot_dir / "report6_patching_comparison.png", saved_paths)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    log_file = Path(cfg.output.log_dir) / "report6_plotting.log"
    setup_logging(level=getattr(cfg.logging, "level", "INFO"), log_file=log_file)

    import logging

    logger = logging.getLogger(__name__)

    summary_dir = Path(getattr(cfg.report6, "summaries_dir", "outputs/report6/summaries"))
    plot_dir = ensure_dir(cfg.output.plots_dir)
    trace_dir = Path(cfg.tracing.output_dir)

    logger.info("Generating Report 6 plots from %s", summary_dir)

    generation_summary = _load_summary_csv(summary_dir, "generation_refusal_rates_by_condition.csv")
    token_summary = _load_summary_csv(summary_dir, "trace_projection_by_condition_layer_token.csv")
    prompt_summary = _load_summary_csv(summary_dir, "trace_prompt_level_key_layers.csv")
    patch_summary = _load_summary_csv(summary_dir, "patching_refusal_recovery_summary.csv")
    raw_trace_df = _load_trace_df(trace_dir)

    saved_paths: list[Path] = []
    order = _condition_order(cfg)

    plot_refusal_rate(generation_summary, order, plot_dir, saved_paths)
    plot_heatmap_comparison(token_summary, cfg, plot_dir, saved_paths)
    plot_layer_line(
        token_summary,
        layer=27,
        title="Report 6: Layer 27 Projection vs Token Position",
        filename="report6_projection_vs_token_layer27.png",
        cfg=cfg,
        plot_dir=plot_dir,
        saved_paths=saved_paths,
    )
    plot_layer_line(
        token_summary,
        layer=int(getattr(cfg.report6, "comparison_layer", 16)),
        title=f"Report 6: Layer {int(getattr(cfg.report6, 'comparison_layer', 16))} Projection vs Token Position",
        filename=f"report6_projection_vs_token_layer{int(getattr(cfg.report6, 'comparison_layer', 16))}.png",
        cfg=cfg,
        plot_dir=plot_dir,
        saved_paths=saved_paths,
    )
    plot_prompt_level_raw(prompt_summary, raw_trace_df, cfg, plot_dir, saved_paths)
    plot_patching(patch_summary, plot_dir, saved_paths)

    if not saved_paths:
        raise FileNotFoundError(
            "No Report 6 plots were generated. Run scripts/summarize_report6_results.py first."
        )

    for path in saved_paths:
        logger.info("Saved plot: %s", path)
        print(path)


if __name__ == "__main__":
    main()
