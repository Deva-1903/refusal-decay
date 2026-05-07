#!/usr/bin/env python3
"""
Generate simple Report 7 plots from summary CSVs.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Report 7 results.")
    parser.add_argument("--summary-dir", type=str, default="outputs/report7/summaries")
    parser.add_argument("--plot-dir", type=str, default="outputs/report7/plots")
    parser.add_argument("--fallback-report6-summary-dir", type=str, default="outputs/report6/summaries")
    return parser.parse_args()


def _read_csv(name: str, summary_dir: Path, fallback_dir: Path | None = None) -> pd.DataFrame | None:
    path = summary_dir / name
    if path.exists():
        return pd.read_csv(path)
    if fallback_dir is not None:
        fallback = fallback_dir / name
        if fallback.exists():
            return pd.read_csv(fallback)
    return None


def _save(fig: plt.Figure, path: Path, saved: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def plot_refusal_rate(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    order = ["harmful_k00", "harmful_k03", "benign_k00", "harmful_k10"]
    df = df.copy()
    df["sort"] = df["condition_name"].map(lambda value: order.index(value) if value in order else 999)
    df = df.sort_values("sort")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#b91c1c" if name.startswith("harmful") else "#2563eb" for name in df["condition_name"]]
    ax.bar(df["condition_name"], df["refusal_rate"] * 100, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Report 7: Refusal Rate by Condition")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, plot_dir / "report7_refusal_rate_by_condition.png", saved)


def plot_generated_mean_by_layer(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in sorted(df["condition"].unique()):
        sub = df[df["condition"] == condition].sort_values("layer")
        ax.plot(sub["layer"], sub["mean_projection"], marker="o", linewidth=2, label=condition)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean generated-token projection")
    ax.set_title("Report 7: Generated Mean Projection by Layer")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    _save(fig, plot_dir / "report7_generated_mean_projection_by_layer.png", saved)


def plot_trajectory(df: pd.DataFrame | None, layer: int, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    sub_df = df[df["layer"] == layer]
    if sub_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition in sorted(sub_df["condition"].unique()):
        sub = sub_df[sub_df["condition"] == condition].sort_values("token_position")
        ax.plot(sub["token_position"], sub["mean_projection"], linewidth=2, label=condition)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Mean projection")
    ax.set_title(f"Report 7: Layer {layer} Projection Trajectory")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    _save(fig, plot_dir / f"report7_layer{layer}_projection_trajectory.png", saved)


def plot_label_association(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    layer = 27
    sub = df[df["layer"] == layer].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sub["label_group"], sub["mean_projection"], color=["#2563eb", "#b91c1c"], edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean generated-token projection")
    ax.set_title("Report 7: Layer 27 Projection by Attacked Label")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, plot_dir / "report7_layer27_projection_by_label.png", saved)


def plot_cross_condition(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    df = df.sort_values(["layer", "target_position"])
    labels = [f"L{int(r.layer)} t{int(r.target_position)}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(df)), df["restoration_rate"] * 100, color="#2563eb", edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Restoration rate (%)")
    ax.set_title("Report 7: Cross-Condition Patching Recovery")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, plot_dir / "report7_cross_condition_patching_recovery.png", saved)


def plot_additive(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    sub = df[(df["direction_type"] == "refusal") & (df["layer"].isin([24, 27]))].copy()
    if sub.empty:
        return
    grouped = (
        sub.groupby(["layer", "alpha"])["restoration_rate"]
        .mean()
        .reset_index()
        .sort_values(["layer", "alpha"])
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for layer in sorted(grouped["layer"].unique()):
        layer_df = grouped[grouped["layer"] == layer]
        ax.plot(layer_df["alpha"], layer_df["restoration_rate"] * 100, marker="o", linewidth=2, label=f"layer {layer}")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean restoration rate (%)")
    ax.set_title("Report 7: Additive Direction Recovery by Alpha")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    _save(fig, plot_dir / "report7_additive_direction_recovery_by_alpha.png", saved)


def plot_comparison(df: pd.DataFrame | None, plot_dir: Path, saved: list[Path]) -> None:
    if df is None or df.empty:
        return
    sub = df[df["layer"].isin([24, 27])].copy()
    if sub.empty:
        return
    labels = [f"{row.method}\nL{int(row.layer)} t{int(row.target_position)}" for row in sub.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(sub)), sub["restoration_rate"] * 100, color="#64748b", edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Restoration rate (%)")
    ax.set_title("Report 7: Report 6 vs Report 7 Intervention Comparison")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, plot_dir / "report7_report6_vs_report7_intervention_comparison.png", saved)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    fallback_dir = Path(args.fallback_report6_summary_dir)
    plot_dir = ensure_dir(args.plot_dir)
    saved: list[Path] = []

    plot_refusal_rate(_read_csv("generation_refusal_rates_by_condition.csv", summary_dir, fallback_dir), plot_dir, saved)
    plot_generated_mean_by_layer(_read_csv("trace_generated_token_mean_by_condition_layer.csv", summary_dir), plot_dir, saved)
    trajectory = _read_csv("trace_token_trajectory_by_condition_layer.csv", summary_dir)
    plot_trajectory(trajectory, 27, plot_dir, saved)
    plot_trajectory(trajectory, 24, plot_dir, saved)
    plot_label_association(_read_csv("prompt_projection_label_association.csv", summary_dir), plot_dir, saved)
    plot_cross_condition(_read_csv("cross_condition_patching_summary.csv", Path("outputs/report7/patching")), plot_dir, saved)
    plot_additive(_read_csv("additive_direction_summary.csv", Path("outputs/report7/interventions")), plot_dir, saved)
    plot_comparison(_read_csv("report6_vs_report7_intervention_comparison.csv", summary_dir), plot_dir, saved)

    if not saved:
        raise FileNotFoundError("No Report 7 plots were generated. Run summaries/interventions first.")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
