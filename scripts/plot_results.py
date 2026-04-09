#!/usr/bin/env python3
"""
Plotting utilities for refusal-decay experiments.

Generates:
  1. refusal_rate_vs_k.png       — line plot of refusal rate as a function of prefix k
  2. projection_vs_position.png  — line plot of refusal projection over token position (per layer)
  3. projection_heatmap.png      — heatmap of projection over (layer × token position)
  4. patching_comparison.png     — bar chart of label changes before/after patching
  5. layer27_prompt_projection_strip.png      — prompt-level strip plot of mean projection by k
  6. layer27_prompt_projection_delta_k00_k03.png — per-prompt paired delta plot for k=0→3
  7. behavioral_prompt_flip_heatmap.png       — prompt-level refusal/compliance heatmap by k

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --sweep-dir outputs/generations/prefilling_sweep
    python scripts/plot_results.py --trace-dir outputs/traces --plot-dir outputs/plots
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir, load_jsonl

setup_logging()
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _load_trace_dataframe(trace_dir: Path) -> pd.DataFrame | None:
    trace_path = trace_dir / "traces_all.parquet"
    if trace_path.exists():
        return pd.read_parquet(trace_path)

    trace_path = trace_dir / "traces_all.csv"
    if trace_path.exists():
        return pd.read_csv(trace_path)

    logger.warning("No combined trace file found in %s", trace_dir)
    return None


def _load_sweep_dataframe(sweep_dir: Path) -> pd.DataFrame | None:
    classified_path = sweep_dir / "sweep_classified.jsonl"
    if classified_path.exists():
        return pd.DataFrame(load_jsonl(classified_path))

    logger.warning("No sweep classified JSONL found in %s", sweep_dir)
    return None


# ---------------------------------------------------------------------------
# Plot 1: Refusal rate vs k
# ---------------------------------------------------------------------------

def plot_refusal_rate_vs_k(sweep_dir: Path, plot_dir: Path) -> None:
    summary_path = sweep_dir / "refusal_rate_summary.csv"
    if not summary_path.exists():
        # Try to build from classified JSONL
        classified_path = sweep_dir / "sweep_classified.jsonl"
        if not classified_path.exists():
            logger.warning("No sweep summary or classified JSONL found in %s", sweep_dir)
            return
        records = load_jsonl(classified_path)
        df_all = pd.DataFrame(records)
        rows = []
        for k, group in df_all.groupby("prefix_k"):
            n_ref = (group["refusal_phrase_label"] == "refusal").sum()
            rows.append({"k": k, "n_prompts": len(group), "n_refusals": n_ref,
                         "refusal_rate": n_ref / len(group)})
        df = pd.DataFrame(rows).sort_values("k")
    else:
        df = pd.read_csv(summary_path).sort_values("k")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k"], df["refusal_rate"] * 100, marker="o", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("Prefix length k (forced tokens)", fontsize=12)
    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_title("Refusal Rate vs. Prefilling Length", fontsize=13)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.grid(alpha=0.3)
    savefig(fig, plot_dir / "refusal_rate_vs_k.png")


# ---------------------------------------------------------------------------
# Plot 2: Projection vs token position (per layer, averaged over prompts)
# ---------------------------------------------------------------------------

def plot_projection_vs_position(trace_dir: Path, plot_dir: Path, max_pos: int = 30) -> None:
    df = _load_trace_dataframe(trace_dir)
    if df is None:
        return

    # Keep only generated token positions (not prefill)
    df = df[~df["is_prefill"]].copy()
    df = df[df["gen_token_pos"] < max_pos]

    layers = sorted(df["layer"].unique())
    k_values = sorted(df["prefix_k"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for layer in layers:
        fig, ax = plt.subplots(figsize=(8, 4))
        for ki, k in enumerate(k_values):
            sub = df[(df["layer"] == layer) & (df["prefix_k"] == k)]
            avg = sub.groupby("gen_token_pos")["projection"].mean()
            ax.plot(avg.index, avg.values, label=f"k={k}", color=colors[ki], linewidth=2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Generated token position", fontsize=11)
        ax.set_ylabel("Refusal direction projection", fontsize=11)
        ax.set_title(f"Refusal Projection vs. Token Position — Layer {layer}", fontsize=12)
        ax.legend(fontsize=9, ncol=len(k_values))
        ax.grid(alpha=0.3)
        savefig(fig, plot_dir / f"projection_vs_position_layer{layer}.png")


# ---------------------------------------------------------------------------
# Plot 3: Heatmap (layer × token position) for a single k value
# ---------------------------------------------------------------------------

def plot_projection_heatmap(trace_dir: Path, plot_dir: Path, k: int = 0, max_pos: int = 30) -> None:
    df = _load_trace_dataframe(trace_dir)
    if df is None:
        logger.warning("No trace file for heatmap.")
        return

    df = df[~df["is_prefill"] & (df["prefix_k"] == k) & (df["gen_token_pos"] < max_pos)]
    if df.empty:
        logger.warning("No data for heatmap with k=%d", k)
        return

    pivot = df.groupby(["layer", "gen_token_pos"])["projection"].mean().unstack(fill_value=0)
    layers = sorted(pivot.index)
    pivot = pivot.loc[layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-abs(pivot.values).max(), vmax=abs(pivot.values).max())
    plt.colorbar(im, ax=ax, label="Refusal direction projection")
    ax.set_xlabel("Generated token position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_title(f"Refusal Projection Heatmap (k={k}) — Layer × Token Position", fontsize=12)
    savefig(fig, plot_dir / f"projection_heatmap_k{k:02d}.png")


# ---------------------------------------------------------------------------
# Plot 4: Patching comparison
# ---------------------------------------------------------------------------

def plot_patching_comparison(patch_dir: Path, plot_dir: Path) -> None:
    classified_path = patch_dir / "patching_classified.jsonl"
    if not classified_path.exists():
        logger.warning("No patching classified results at %s", classified_path)
        return

    records = load_jsonl(classified_path)
    df = pd.DataFrame([r for r in records if "error" not in r])

    if df.empty:
        logger.warning("No valid patching records.")
        return

    # Phrase-label refusal rate before and after, grouped by (layer, target_position).
    # Expected direction: patched_refusal > baseline_refusal (patching restores refusal signal).
    groups = df.groupby(["layer", "target_position"]).agg(
        baseline_refusal=("baseline_phrase_label", lambda x: (x == "refusal").mean()),
        patched_refusal=("patched_phrase_label", lambda x: (x == "refusal").mean()),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    width = 0.35
    labels = [f"L{row.layer}\ntt={row.target_position}" for _, row in groups.iterrows()]

    ax.bar(x - width/2, groups["baseline_refusal"] * 100, width, label="Baseline", color="steelblue")
    ax.bar(x + width/2, groups["patched_refusal"] * 100, width, label="Patched", color="salmon")

    ax.set_xlabel("(Layer, Target Position)", fontsize=11)
    ax.set_ylabel("Refusal rate (%)", fontsize=11)
    ax.set_title("Refusal Rate Before/After Direction Patching\n(↑ patched = expected causal effect)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, plot_dir / "patching_comparison.png")


# ---------------------------------------------------------------------------
# Plot 5: Prompt-level strip plot for one layer
# ---------------------------------------------------------------------------

def plot_prompt_level_projection_strip(
    trace_dir: Path,
    sweep_dir: Path,
    plot_dir: Path,
    layer: int = 27,
    max_pos: int = 30,
) -> None:
    trace_df = _load_trace_dataframe(trace_dir)
    if trace_df is None:
        return

    trace_df = trace_df[
        (~trace_df["is_prefill"])
        & (trace_df["layer"] == layer)
        & (trace_df["gen_token_pos"] < max_pos)
    ].copy()
    if trace_df.empty:
        logger.warning("No prompt-level trace data for layer %d", layer)
        return

    prompt_means = (
        trace_df.groupby(["prompt_id", "prefix_k"])["projection"]
        .mean()
        .reset_index(name="mean_projection")
    )

    sweep_df = _load_sweep_dataframe(sweep_dir)
    if sweep_df is not None and not sweep_df.empty:
        labels = sweep_df[["prompt_id", "prefix_k", "refusal_phrase_label"]].drop_duplicates()
        prompt_means = prompt_means.merge(labels, on=["prompt_id", "prefix_k"], how="left")
    else:
        prompt_means["refusal_phrase_label"] = "unknown"

    k_values = sorted(prompt_means["prefix_k"].unique())
    x_lookup = {k: i for i, k in enumerate(k_values)}
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.18, 0.18, size=len(prompt_means))
    color_map = {
        "refusal": "#B91C1C",
        "compliance": "#2563EB",
        "unknown": "#6B7280",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, row in prompt_means.reset_index(drop=True).iterrows():
        x = x_lookup[row["prefix_k"]] + jitter[idx]
        color = color_map.get(row["refusal_phrase_label"], "#6B7280")
        ax.scatter(
            x,
            row["mean_projection"],
            s=44,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    summary = (
        prompt_means.groupby("prefix_k")["mean_projection"]
        .agg(["mean", "median"])
        .reset_index()
    )
    ax.plot(
        [x_lookup[k] for k in summary["prefix_k"]],
        summary["mean"],
        color="black",
        linewidth=1.5,
        marker="D",
        markersize=6,
        label="Mean",
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_xlabel("Prefilling length", fontsize=11)
    ax.set_ylabel(f"Mean projection at layer {layer}\n(avg. over generated positions 0-{max_pos-1})", fontsize=11)
    ax.set_title(
        f"Prompt-Level Refusal Projection at Layer {layer}\nEach point is one harmful prompt; color = behavioral label",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    legend_handles = [
        Patch(facecolor=color_map["refusal"], label="Refusal"),
        Patch(facecolor=color_map["compliance"], label="Compliance"),
        plt.Line2D([0], [0], color="black", marker="D", linewidth=1.5, markersize=6, label="Mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=True)
    savefig(fig, plot_dir / f"layer{layer}_prompt_projection_strip.png")


# ---------------------------------------------------------------------------
# Plot 6: Prompt-level paired delta plot for k=0 -> k=3
# ---------------------------------------------------------------------------

def plot_prompt_level_delta_k00_k03(
    trace_dir: Path,
    sweep_dir: Path,
    plot_dir: Path,
    layer: int = 27,
    max_pos: int = 30,
) -> None:
    trace_df = _load_trace_dataframe(trace_dir)
    if trace_df is None:
        return

    trace_df = trace_df[
        (~trace_df["is_prefill"])
        & (trace_df["layer"] == layer)
        & (trace_df["gen_token_pos"] < max_pos)
        & (trace_df["prefix_k"].isin([0, 3]))
    ].copy()
    if trace_df.empty:
        logger.warning("No k=0/3 trace data for layer %d", layer)
        return

    prompt_means = (
        trace_df.groupby(["prompt_id", "prefix_k"])["projection"]
        .mean()
        .reset_index(name="mean_projection")
    )
    wide = prompt_means.pivot(index="prompt_id", columns="prefix_k", values="mean_projection").dropna()
    if wide.empty:
        logger.warning("No paired prompt means available for k=0 and k=3")
        return

    sweep_df = _load_sweep_dataframe(sweep_dir)
    if sweep_df is not None and not sweep_df.empty:
        labels = sweep_df[["prompt_id", "prefix_k", "refusal_phrase_label"]].drop_duplicates()
        label_wide = labels.pivot(index="prompt_id", columns="prefix_k", values="refusal_phrase_label")
        wide = wide.join(label_wide.add_prefix("label_k"))

    wide["delta_0_to_3"] = wide[3] - wide[0]
    wide = wide.sort_values("delta_0_to_3")

    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(wide))
    for yi, (_, row) in enumerate(wide.iterrows()):
        color = "#6B7280"
        if row.get("label_k0") == "refusal" and row.get("label_k3") == "compliance":
            color = "#DC2626"
        elif row.get("label_k0") == row.get("label_k3"):
            color = "#2563EB"
        ax.plot([row[0], row[3]], [yi, yi], color=color, linewidth=1.4, alpha=0.8)
        ax.scatter(row[0], yi, color="#111827", s=22, zorder=3)
        ax.scatter(row[3], yi, color=color, s=22, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index.tolist(), fontsize=7)
    ax.set_xlabel(f"Mean projection at layer {layer} (generated positions 0-{max_pos-1})", fontsize=11)
    ax.set_ylabel("Prompt ID (sorted by k=3 - k=0 shift)", fontsize=11)
    ax.set_title(
        f"Per-Prompt Shift in Refusal Projection at Layer {layer}\nPaired values for k=0 and k=3",
        fontsize=12,
    )
    legend_handles = [
        Patch(facecolor="#DC2626", label="Refusal -> Compliance"),
        Patch(facecolor="#2563EB", label="No label flip"),
        Patch(facecolor="#6B7280", label="Other pattern"),
        plt.Line2D([0], [0], color="#111827", marker="o", linewidth=0, markersize=5, label="k=0 point"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.25)
    savefig(fig, plot_dir / f"layer{layer}_prompt_projection_delta_k00_k03.png")


# ---------------------------------------------------------------------------
# Plot 7: Prompt-level behavioral heatmap by k
# ---------------------------------------------------------------------------

def plot_behavioral_prompt_flip_heatmap(sweep_dir: Path, plot_dir: Path) -> None:
    sweep_df = _load_sweep_dataframe(sweep_dir)
    if sweep_df is None or sweep_df.empty:
        return

    df = sweep_df[sweep_df["prefix_k"].isin([0, 3, 10])].copy()
    df["behavior_code"] = (df["refusal_phrase_label"] == "refusal").astype(int)

    pivot = df.pivot(index="prompt_id", columns="prefix_k", values="behavior_code")
    if pivot.empty:
        logger.warning("No behavioral prompt-level data available")
        return

    sort_cols = [k for k in [0, 3, 10] if k in pivot.columns]
    pivot = pivot.sort_values(by=sort_cols, ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 8))
    cmap = ListedColormap(["#2563EB", "#B91C1C"])
    ax.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"k={k}" for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    ax.set_xlabel("Prefilling length", fontsize=11)
    ax.set_ylabel("Prompt ID", fontsize=11)
    ax.set_title("Behavioral Outcome by Prompt and Prefilling Length\nRed = refusal, blue = compliance", fontsize=12)
    legend_handles = [
        Patch(facecolor="#B91C1C", label="Refusal"),
        Patch(facecolor="#2563EB", label="Compliance"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True)
    savefig(fig, plot_dir / "behavioral_prompt_flip_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all result plots.")
    parser.add_argument("--sweep-dir", type=str, default="outputs/generations/prefilling_sweep")
    parser.add_argument("--trace-dir", type=str, default="outputs/traces")
    parser.add_argument("--patch-dir", type=str, default="outputs/patches")
    parser.add_argument("--plot-dir", type=str, default="outputs/plots")
    parser.add_argument("--k-heatmap", type=int, default=0,
                        help="Which k value to use for the heatmap plot.")
    parser.add_argument("--prompt-layer", type=int, default=27,
                        help="Which layer to use for prompt-level projection plots.")
    parser.add_argument("--max-pos", type=int, default=30,
                        help="Use generated positions [0, max_pos) for prompt-level summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dir = ensure_dir(args.plot_dir)

    plot_refusal_rate_vs_k(Path(args.sweep_dir), plot_dir)
    plot_projection_vs_position(Path(args.trace_dir), plot_dir, max_pos=args.max_pos)
    plot_projection_heatmap(Path(args.trace_dir), plot_dir, k=args.k_heatmap, max_pos=args.max_pos)
    plot_patching_comparison(Path(args.patch_dir), plot_dir)
    plot_prompt_level_projection_strip(
        Path(args.trace_dir),
        Path(args.sweep_dir),
        plot_dir,
        layer=args.prompt_layer,
        max_pos=args.max_pos,
    )
    plot_prompt_level_delta_k00_k03(
        Path(args.trace_dir),
        Path(args.sweep_dir),
        plot_dir,
        layer=args.prompt_layer,
        max_pos=args.max_pos,
    )
    plot_behavioral_prompt_flip_heatmap(Path(args.sweep_dir), plot_dir)

    logger.info("All plots saved to %s", plot_dir)


if __name__ == "__main__":
    main()
