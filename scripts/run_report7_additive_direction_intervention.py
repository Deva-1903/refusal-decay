#!/usr/bin/env python3
"""
Report 7 additive direction intervention.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.classification.refusal_classifier import build_classifier_from_config
from src.config import load_config, save_config_snapshot
from src.data.loader import load_benign_prompts, load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer
from src.patching.report7_interventions import (
    generate_baseline,
    generate_with_additive_direction,
    make_direction_for_type,
)
from src.probing.direction import load_direction
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Report 7 additive direction intervention.")
    parser.add_argument("--config", type=str, default="configs/experiments/report7/additive_intervention_report7.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--direction-path", type=str, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    return parser.parse_args()


def _apply_model_override(cfg, model_config_path: str | None):
    if not model_config_path:
        return cfg
    model_cfg = load_config(model_config_path)
    cfg.model = model_cfg.model
    if hasattr(model_cfg, "probing"):
        cfg.probing = model_cfg.probing
    return cfg


def _truncate(text: str, limit: int = 240) -> str:
    return (text or "")[:limit]


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    valid = df[df["error"].fillna("") == ""].copy()
    if valid.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    # Per-seed aggregation: each (cell, seed) is its own row.
    summary = (
        valid.groupby(["dataset", "condition", "layer", "target_position", "alpha", "direction_type", "seed"])
        .agg(
            n_prompts=("prompt_id", "size"),
            restored=("restored_refusal", "sum"),
            lost=("lost_refusal", "sum"),
            unchanged=("unchanged", "sum"),
        )
        .reset_index()
    )
    summary["restoration_rate"] = summary["restored"] / summary["n_prompts"]
    summary["loss_rate"] = summary["lost"] / summary["n_prompts"]
    summary.to_csv(path, index=False)


def _write_seed_aggregated_summary(df: pd.DataFrame, path: Path) -> None:
    """Aggregate restoration_rate across seeds for the multi-seed random/orthogonal controls."""
    valid = df[df["error"].fillna("") == ""].copy()
    if valid.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    cell_cols = ["dataset", "condition", "layer", "target_position", "alpha", "direction_type"]
    per_seed = (
        valid.groupby(cell_cols + ["seed"])
        .agg(
            n_prompts=("prompt_id", "size"),
            restored=("restored_refusal", "sum"),
            lost=("lost_refusal", "sum"),
        )
        .reset_index()
    )
    per_seed["restoration_rate"] = per_seed["restored"] / per_seed["n_prompts"]
    per_seed["loss_rate"] = per_seed["lost"] / per_seed["n_prompts"]
    aggregated = (
        per_seed.groupby(cell_cols)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_restoration_rate=("restoration_rate", "mean"),
            std_restoration_rate=("restoration_rate", "std"),
            min_restoration_rate=("restoration_rate", "min"),
            max_restoration_rate=("restoration_rate", "max"),
            mean_loss_rate=("loss_rate", "mean"),
            std_loss_rate=("loss_rate", "std"),
        )
        .reset_index()
    )
    aggregated.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)
    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path
    if args.max_prompts is not None:
        cfg.data.max_prompts = args.max_prompts

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "report7_additive_intervention.log",
    )

    import logging

    logger = logging.getLogger(__name__)
    out_dir = ensure_dir(cfg.intervention.output_dir)
    save_config_snapshot(cfg, out_dir / "additive_intervention_config_snapshot.yaml")

    condition = getattr(cfg.report7, "condition", "harmful_k03")
    dataset = getattr(cfg.report7, "dataset", "harmful")
    k = int(cfg.prefilling.k_values[0])
    prefix_text = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    max_new_tokens = int(getattr(cfg.generation, "max_new_tokens", 64))
    layers = [int(layer) for layer in cfg.intervention.layers]
    target_positions = [int(position) for position in cfg.intervention.target_positions]
    alphas = [float(alpha) for alpha in cfg.intervention.alphas]
    direction_types = list(getattr(cfg.intervention, "direction_types", ["refusal"]))
    # Multi-seed support: random and orthogonal directions iterate over `seeds`.
    # `refusal` is deterministic and always uses the first seed only.
    seeds = list(getattr(cfg.intervention, "seeds", [int(getattr(cfg.intervention, "seed", 42))]))
    seeds = [int(s) for s in seeds]

    logger.info("Dataset: %s", dataset)
    logger.info("Condition: %s", condition)
    logger.info("Layers: %s", layers)
    logger.info("Target positions: %s", target_positions)
    logger.info("Alphas: %s", alphas)
    logger.info("Direction types: %s", direction_types)
    logger.info("Seeds: %s", seeds)

    if dataset == "harmful":
        prompts = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=getattr(cfg.data, "max_prompts", None))
    elif dataset == "benign":
        prompts = load_benign_prompts(cfg.data.benign_prompts, max_prompts=getattr(cfg.data, "max_prompts", None))
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}; expected 'harmful' or 'benign'.")
    directions = load_direction(cfg.probing.direction_save_path)
    model, tokenizer = load_model_and_tokenizer(cfg)
    classifier = build_classifier_from_config(cfg)

    baseline_cache: dict[str, tuple[str, list[int], str]] = {}
    rows: list[dict] = []

    for layer in layers:
        if layer not in directions:
            logger.warning("No refusal direction for layer %d; skipping.", layer)
            continue
        for direction_type in direction_types:
            # `refusal` is deterministic — only run once even with multiple seeds.
            seeds_for_type = seeds if direction_type in ("random", "orthogonal") else seeds[:1]
            for current_seed in seeds_for_type:
                intervention_direction = make_direction_for_type(
                    directions[layer], direction_type, seed=current_seed, layer=layer
                )
                for target_position in target_positions:
                    if target_position < 0:
                        logger.warning("Skipping unsupported target_position=%d; generated-token positions must be >= 0.", target_position)
                        continue
                    for alpha in alphas:
                        logger.info(
                            "Running layer=%d target=%d alpha=%.2f direction=%s seed=%d",
                            layer, target_position, alpha, direction_type, current_seed,
                        )
                        for prompt in prompts:
                            row = {
                                "prompt_id": prompt.prompt_id,
                                "dataset": dataset,
                                "condition": condition,
                                "layer": layer,
                                "target_position": target_position,
                                "alpha": alpha,
                                "direction_type": direction_type,
                                "seed": current_seed,
                                "baseline_attacked_label": None,
                                "intervened_label": None,
                                "restored_refusal": False,
                                "lost_refusal": False,
                                "unchanged": False,
                                "baseline_output_truncated": "",
                                "intervened_output_truncated": "",
                                "error": "",
                            }
                            try:
                                if prompt.prompt_id not in baseline_cache:
                                    baseline_text, baseline_ids = generate_baseline(
                                        model, tokenizer, prompt, k, prefix_text, max_new_tokens
                                    )
                                    baseline_cache[prompt.prompt_id] = (
                                        baseline_text,
                                        baseline_ids,
                                        classifier.classify(baseline_text),
                                    )
                                baseline_text, _, baseline_label = baseline_cache[prompt.prompt_id]
                                intervened = generate_with_additive_direction(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt,
                                    k=k,
                                    prefix_text=prefix_text,
                                    layer=layer,
                                    target_position=target_position,
                                    direction=intervention_direction,
                                    alpha=alpha,
                                    max_new_tokens=max_new_tokens,
                                )
                                intervened_label = classifier.classify(intervened.text)
                                restored = baseline_label == "compliance" and intervened_label == "refusal"
                                lost = baseline_label == "refusal" and intervened_label == "compliance"
                                row.update(
                                    baseline_attacked_label=baseline_label,
                                    intervened_label=intervened_label,
                                    restored_refusal=restored,
                                    lost_refusal=lost,
                                    unchanged=not restored and not lost,
                                    baseline_output_truncated=_truncate(baseline_text),
                                    intervened_output_truncated=_truncate(intervened.text),
                                )
                            except Exception as exc:
                                logger.error(
                                    "Error on prompt=%s layer=%d target=%d alpha=%.2f direction=%s seed=%d: %s",
                                    prompt.prompt_id, layer, target_position, alpha, direction_type, current_seed, exc,
                                )
                                row["error"] = f"{type(exc).__name__}: {exc}"
                            rows.append(row)

    df = pd.DataFrame(rows)
    # Output filenames distinguish harmful run from benign positive control.
    suffix = "" if dataset == "harmful" else f"_{dataset}"
    result_path = out_dir / f"additive_direction_results{suffix}.csv"
    summary_path = out_dir / f"additive_direction_summary{suffix}.csv"
    seed_agg_path = out_dir / f"additive_direction_seed_aggregated{suffix}.csv"
    df.to_csv(result_path, index=False)
    _write_summary(df, summary_path)
    _write_seed_aggregated_summary(df, seed_agg_path)
    logger.info("Saved additive intervention results to %s", result_path)
    logger.info("Saved additive intervention summary to %s", summary_path)
    logger.info("Saved seed-aggregated summary to %s", seed_agg_path)
    print(result_path)
    print(summary_path)
    print(seed_agg_path)


if __name__ == "__main__":
    main()
