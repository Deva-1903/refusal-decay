#!/usr/bin/env python3
"""
Plotting utilities for refusal-decay experiments.

Generates:
  1. refusal_rate_vs_k.png       — line plot of refusal rate as a function of prefix k
  2. projection_vs_position.png  — line plot of refusal projection over token position (per layer)
  3. projection_heatmap.png      — heatmap of projection over (layer × token position)
  4. patching_comparison.png     — bar chart of label changes before/after patching

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
            n_ref = (group["refusal_label"] == "refusal").sum()
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
    trace_path = trace_dir / "traces_all.parquet"
    if not trace_path.exists():
        trace_path = trace_dir / "traces_all.csv"
        if not trace_path.exists():
            logger.warning("No combined trace file found in %s", trace_dir)
            return
        df = pd.read_csv(trace_path)
    else:
        df = pd.read_parquet(trace_path)

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
    trace_path = trace_dir / "traces_all.parquet"
    if not trace_path.exists():
        trace_path = trace_dir / "traces_all.csv"
        if not trace_path.exists():
            logger.warning("No trace file for heatmap.")
            return
        df = pd.read_csv(trace_path)
    else:
        df = pd.read_parquet(trace_path)

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

    # Refusal rate before and after, grouped by (layer, target_position)
    groups = df.groupby(["layer", "target_position"]).agg(
        baseline_refusal=("baseline_label", lambda x: (x == "refusal").mean()),
        patched_refusal=("patched_label", lambda x: (x == "refusal").mean()),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    width = 0.35
    labels = [f"L{row.layer}\ntt={row.target_position}" for _, row in groups.iterrows()]

    ax.bar(x - width/2, groups["baseline_refusal"] * 100, width, label="Baseline", color="steelblue")
    ax.bar(x + width/2, groups["patched_refusal"] * 100, width, label="Patched", color="salmon")

    ax.set_xlabel("(Layer, Target Position)", fontsize=11)
    ax.set_ylabel("Refusal rate (%)", fontsize=11)
    ax.set_title("Refusal Rate Before/After Direction Patching", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, plot_dir / "patching_comparison.png")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dir = ensure_dir(args.plot_dir)

    plot_refusal_rate_vs_k(Path(args.sweep_dir), plot_dir)
    plot_projection_vs_position(Path(args.trace_dir), plot_dir)
    plot_projection_heatmap(Path(args.trace_dir), plot_dir, k=args.k_heatmap)
    plot_patching_comparison(Path(args.patch_dir), plot_dir)

    logger.info("All plots saved to %s", plot_dir)


if __name__ == "__main__":
    main()
