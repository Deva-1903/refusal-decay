#!/usr/bin/env python3
"""
Statistical inference for Report 7 intervention cells.

For each (method, layer, target_position[, alpha, direction_type]) cell, compute:
  - bootstrap 95% CI on restoration_rate and loss_rate (per-prompt resampling)
  - McNemar's exact test comparing the per-prompt patched/intervened label
    against the per-prompt attacked-baseline label, when the discordant-pair
    count is at least `--mcnemar-min-discordant` (default 5)

Inputs (per-prompt CSVs already produced by the run scripts):
  - outputs/report7/patching/cross_condition_patching_results.csv
  - outputs/report7/interventions/additive_direction_results.csv

Outputs:
  - outputs/report7/summaries/cross_condition_patching_stats.csv
  - outputs/report7/summaries/additive_direction_stats.csv
  - outputs/report7/summaries/report7_intervention_stats_combined.csv

Notes:
  - Bootstrap is over per-prompt outcome 0/1 vectors with B resamples.
  - McNemar uses the discordant counts (b = patched_refusal & baseline_compliance,
    c = patched_compliance & baseline_refusal) and the exact binomial p-value.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir


CROSS_GROUP_COLS = ["source_condition", "target_condition", "layer", "source_position", "target_position", "mode"]
# Additive groups by `seed` so each (cell, seed) gets its own CIs/McNemar.
# Aggregation across seeds for the random/orthogonal controls is in the
# `additive_direction_seed_aggregated*.csv` file produced by the run script.
ADDITIVE_GROUP_COLS = ["dataset", "condition", "layer", "target_position", "alpha", "direction_type", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap + McNemar stats for R7 interventions.")
    parser.add_argument("--cross-results", type=str, default="outputs/report7/patching/cross_condition_patching_results.csv")
    parser.add_argument("--additive-results", type=str, default="outputs/report7/interventions/additive_direction_results.csv")
    parser.add_argument(
        "--additive-benign-results",
        type=str,
        default="outputs/report7/interventions_benign_control/additive_direction_results_benign.csv",
        help="Benign positive-control additive results; processed in addition to harmful.",
    )
    parser.add_argument("--summary-dir", type=str, default="outputs/report7/summaries")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--mcnemar-min-discordant", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def _bootstrap_rate_ci(values: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a 0/1 vector. Returns (rate, lo, hi)."""
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rate = float(values.mean())
    if n == 1:
        return rate, rate, rate
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return rate, lo, hi


def _binomial_two_sided_pvalue(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar with discordant counts b, c."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), times 2, capped at 1.
    cum = 0.0
    for i in range(0, k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2.0 * cum)
    return p


def _mcnemar(baseline_labels: pd.Series, intervened_labels: pd.Series, min_discordant: int) -> dict:
    """McNemar on per-prompt paired labels. Only computes when discordant pairs >= min."""
    valid = baseline_labels.isin(["refusal", "compliance"]) & intervened_labels.isin(["refusal", "compliance"])
    base = baseline_labels[valid].reset_index(drop=True)
    interv = intervened_labels[valid].reset_index(drop=True)
    n = len(base)
    # b = baseline=compliance & intervened=refusal  (restored)
    # c = baseline=refusal    & intervened=compliance (lost)
    b = int(((base == "compliance") & (interv == "refusal")).sum())
    c = int(((base == "refusal") & (interv == "compliance")).sum())
    discordant = b + c
    if discordant < min_discordant:
        return {
            "mcnemar_n_pairs": n,
            "mcnemar_b_restored": b,
            "mcnemar_c_lost": c,
            "mcnemar_discordant": discordant,
            "mcnemar_p_value": float("nan"),
            "mcnemar_skipped_reason": f"discordant<{min_discordant}",
        }
    p = _binomial_two_sided_pvalue(b, c)
    return {
        "mcnemar_n_pairs": n,
        "mcnemar_b_restored": b,
        "mcnemar_c_lost": c,
        "mcnemar_discordant": discordant,
        "mcnemar_p_value": p,
        "mcnemar_skipped_reason": "",
    }


def _compute_cell_stats(group: pd.DataFrame, n_boot: int, ci: float, rng: np.random.Generator,
                        baseline_col: str, intervened_col: str, min_discordant: int) -> dict:
    restored = group["restored_refusal"].astype(int).to_numpy()
    lost = group["lost_refusal"].astype(int).to_numpy()
    rest_rate, rest_lo, rest_hi = _bootstrap_rate_ci(restored, n_boot, ci, rng)
    loss_rate, loss_lo, loss_hi = _bootstrap_rate_ci(lost, n_boot, ci, rng)
    out = {
        "n_prompts": int(group.shape[0]),
        "restored": int(restored.sum()),
        "lost": int(lost.sum()),
        "unchanged": int(group["unchanged"].astype(int).sum()),
        "restoration_rate": rest_rate,
        "restoration_ci_lo": rest_lo,
        "restoration_ci_hi": rest_hi,
        "loss_rate": loss_rate,
        "loss_ci_lo": loss_lo,
        "loss_ci_hi": loss_hi,
    }
    out.update(_mcnemar(group[baseline_col], group[intervened_col], min_discordant))
    return out


def _compute_stats(df: pd.DataFrame, group_cols: list[str], baseline_col: str, intervened_col: str,
                   n_boot: int, ci: float, rng: np.random.Generator, min_discordant: int) -> pd.DataFrame:
    valid = df[df["error"].fillna("") == ""].copy()
    if valid.empty:
        return pd.DataFrame()
    # Drop group cols that are missing from this dataframe (older CSVs may
    # predate the dataset/seed columns added in the multi-seed upgrade).
    available = [col for col in group_cols if col in valid.columns]
    rows: list[dict] = []
    for keys, group in valid.groupby(available):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cell = dict(zip(available, keys))
        cell.update(_compute_cell_stats(group, n_boot, ci, rng, baseline_col, intervened_col, min_discordant))
        rows.append(cell)
    return pd.DataFrame(rows)


def _aggregate_additive_seed_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed cell stats into a per-cell mean +/- std table.

    Useful for random/orthogonal controls where the cell exists at multiple
    seeds. `refusal` direction-type cells with a single seed pass through
    with std = 0 / NaN.
    """
    if stats.empty or "seed" not in stats.columns:
        return pd.DataFrame()
    cell_cols = [col for col in ["dataset", "condition", "layer", "target_position", "alpha", "direction_type"] if col in stats.columns]
    aggregated = (
        stats.groupby(cell_cols)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_restoration_rate=("restoration_rate", "mean"),
            std_restoration_rate=("restoration_rate", "std"),
            min_restoration_rate=("restoration_rate", "min"),
            max_restoration_rate=("restoration_rate", "max"),
            mean_loss_rate=("loss_rate", "mean"),
            mean_ci_lo=("restoration_ci_lo", "mean"),
            mean_ci_hi=("restoration_ci_hi", "mean"),
        )
        .reset_index()
    )
    return aggregated


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    summary_dir = ensure_dir(args.summary_dir)

    cross_path = Path(args.cross_results)
    additive_path = Path(args.additive_results)

    cross_stats = pd.DataFrame()
    additive_stats = pd.DataFrame()

    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)
        cross_stats = _compute_stats(
            cross_df, CROSS_GROUP_COLS,
            baseline_col="baseline_attacked_label",
            intervened_col="patched_label",
            n_boot=args.n_boot, ci=args.ci, rng=rng, min_discordant=args.mcnemar_min_discordant,
        )
        cross_stats["method"] = "report7_baseline_source_cross_condition"
        cross_out = summary_dir / "cross_condition_patching_stats.csv"
        cross_stats.to_csv(cross_out, index=False)
        print(cross_out)
    else:
        print(f"WARN: missing {cross_path} — skipping cross-condition stats")

    if additive_path.exists():
        additive_df = pd.read_csv(additive_path)
        additive_stats = _compute_stats(
            additive_df, ADDITIVE_GROUP_COLS,
            baseline_col="baseline_attacked_label",
            intervened_col="intervened_label",
            n_boot=args.n_boot, ci=args.ci, rng=rng, min_discordant=args.mcnemar_min_discordant,
        )
        if not additive_stats.empty:
            additive_stats["method"] = additive_stats.apply(
                lambda r: f"report7_additive_{r['direction_type']}_alpha_{float(r['alpha']):g}", axis=1,
            )
            additive_out = summary_dir / "additive_direction_stats.csv"
            additive_stats.to_csv(additive_out, index=False)
            print(additive_out)

            additive_seed_agg = _aggregate_additive_seed_stats(additive_stats)
            if not additive_seed_agg.empty:
                additive_seed_agg_out = summary_dir / "additive_direction_stats_seed_aggregated.csv"
                additive_seed_agg.to_csv(additive_seed_agg_out, index=False)
                print(additive_seed_agg_out)
        else:
            print(f"WARN: {additive_path} has no non-error rows; additive cells all failed — check the dtype/run-script setup")
    else:
        print(f"WARN: missing {additive_path} — skipping additive stats")

    benign_path = Path(args.additive_benign_results)
    benign_stats = pd.DataFrame()
    if benign_path.exists():
        benign_df = pd.read_csv(benign_path)
        benign_stats = _compute_stats(
            benign_df, ADDITIVE_GROUP_COLS,
            baseline_col="baseline_attacked_label",
            intervened_col="intervened_label",
            n_boot=args.n_boot, ci=args.ci, rng=rng, min_discordant=args.mcnemar_min_discordant,
        )
        if not benign_stats.empty:
            benign_stats["method"] = benign_stats.apply(
                lambda r: f"report7_additive_benign_{r['direction_type']}_alpha_{float(r['alpha']):g}", axis=1,
            )
            benign_out = summary_dir / "additive_direction_stats_benign_control.csv"
            benign_stats.to_csv(benign_out, index=False)
            print(benign_out)
            benign_seed_agg = _aggregate_additive_seed_stats(benign_stats)
            if not benign_seed_agg.empty:
                benign_seed_agg_out = summary_dir / "additive_direction_stats_seed_aggregated_benign_control.csv"
                benign_seed_agg.to_csv(benign_seed_agg_out, index=False)
                print(benign_seed_agg_out)
        else:
            print(f"WARN: {benign_path} has no non-error rows; benign control cells all failed")
    else:
        print(f"INFO: no benign positive-control results at {benign_path} (run additive_intervention_benign_control_report7.yaml to populate)")

    combined = pd.concat([cross_stats, additive_stats, benign_stats], ignore_index=True, sort=False)
    if not combined.empty:
        combined_out = summary_dir / "report7_intervention_stats_combined.csv"
        combined.to_csv(combined_out, index=False)
        print(combined_out)


if __name__ == "__main__":
    main()
