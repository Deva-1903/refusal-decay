#!/usr/bin/env python3
"""
Run an all-position directional intervention (ablate or add) and record the
behavioral change versus an unintervened baseline.

Covers both paper experiments with one runner:
  - E1 (positive control): clean setting (k=0). Ablating the direction on
    harmful prompts should drop refusal; adding it on benign prompts should
    induce refusal. This proves the intervention can move behavior.
  - E2 (headline contrast): attacked setting (k=3, k=10). The same
    intervention is expected NOT to restore refusal.

Interpretation is intentionally left to analysis: this script records the
per-prompt baseline and intervened labels plus transition counts and both
refusal rates, so each experiment is read against its own expectation.
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
from src.patching.directional_intervention import (
    broadcast_direction,
    generate_text,
    generate_with_directional_intervention,
    per_layer_directions,
)
from src.probing.direction import load_direction
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an all-position directional intervention.")
    parser.add_argument("--config", type=str, required=True)
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


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)
    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path
    if args.max_prompts is not None:
        cfg.data.max_prompts = args.max_prompts

    paper = getattr(cfg, "paper", object())
    experiment_name = getattr(paper, "experiment_name", "directional_intervention")
    dataset = getattr(paper, "dataset", "harmful")

    out_dir = ensure_dir(getattr(paper, "output_dir", "outputs/paper/directional"))
    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / f"{experiment_name}.log",
    )
    import logging

    logger = logging.getLogger(__name__)
    save_config_snapshot(cfg, out_dir / f"{experiment_name}_config_snapshot.yaml")

    k = int(getattr(cfg.prefilling, "k", 0))
    prefix_text = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    max_new_tokens = int(getattr(cfg.generation, "max_new_tokens", 64))

    mode = getattr(cfg.intervention, "mode", "ablate")
    alpha = float(getattr(cfg.intervention, "alpha", 1.0))
    strategy = getattr(cfg.intervention, "direction_strategy", "broadcast")

    if dataset == "harmful":
        prompts = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=getattr(cfg.data, "max_prompts", None))
    elif dataset == "benign":
        prompts = load_benign_prompts(cfg.data.benign_prompts, max_prompts=getattr(cfg.data, "max_prompts", None))
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    directions = load_direction(cfg.probing.direction_save_path)
    model, tokenizer = load_model_and_tokenizer(cfg)
    classifier = build_classifier_from_config(cfg)

    # Resolve layers after the model is loaded so "all" can expand to the
    # model's true layer count (works across 8B/3B/Qwen without per-model edits).
    raw_layers = cfg.intervention.layers
    if isinstance(raw_layers, str) and raw_layers.lower() == "all":
        layers = list(range(model.config.num_hidden_layers))
    else:
        layers = [int(layer) for layer in raw_layers]
    source_layer = int(getattr(cfg.intervention, "source_layer", layers[-1]))

    logger.info("Experiment: %s | dataset=%s k=%d mode=%s alpha=%.2f", experiment_name, dataset, k, mode, alpha)
    logger.info("Layers=%d hooked strategy=%s source_layer=%d", len(layers), strategy, source_layer)

    if strategy == "broadcast":
        if source_layer not in directions:
            raise ValueError(f"source_layer {source_layer} has no extracted direction.")
        layer_directions = broadcast_direction(directions[source_layer], layers)
    elif strategy == "per_layer":
        layer_directions = per_layer_directions(directions, layers)
    else:
        raise ValueError(f"Unknown direction_strategy: {strategy!r}")

    rows: list[dict] = []
    for i, prompt in enumerate(prompts):
        row = {
            "experiment": experiment_name,
            "prompt_id": prompt.prompt_id,
            "dataset": dataset,
            "k": k,
            "mode": mode,
            "alpha": alpha,
            "strategy": strategy,
            "source_layer": source_layer,
            "n_layers": len(layer_directions),
            "baseline_label": None,
            "intervened_label": None,
            "baseline_to_refusal": False,
            "baseline_to_compliance": False,
            "unchanged": False,
            "baseline_output_truncated": "",
            "intervened_output_truncated": "",
            "error": "",
        }
        try:
            baseline_text = generate_text(model, tokenizer, prompt, k, prefix_text, max_new_tokens)
            baseline_label = classifier.classify(baseline_text)
            intervened_text = generate_with_directional_intervention(
                model, tokenizer, prompt, k, prefix_text, layer_directions, mode, alpha, max_new_tokens
            )
            intervened_label = classifier.classify(intervened_text)
            row.update(
                baseline_label=baseline_label,
                intervened_label=intervened_label,
                baseline_to_refusal=(baseline_label == "compliance" and intervened_label == "refusal"),
                baseline_to_compliance=(baseline_label == "refusal" and intervened_label == "compliance"),
                unchanged=(baseline_label == intervened_label),
                baseline_output_truncated=_truncate(baseline_text),
                intervened_output_truncated=_truncate(intervened_text),
            )
        except Exception as exc:
            logger.error("Error on prompt=%s: %s", prompt.prompt_id, exc)
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d prompts done", i + 1, len(prompts))

    df = pd.DataFrame(rows)
    result_path = out_dir / f"{experiment_name}_results.csv"
    df.to_csv(result_path, index=False)

    valid = df[df["error"].fillna("") == ""]
    summary = {
        "experiment": experiment_name,
        "dataset": dataset,
        "k": k,
        "mode": mode,
        "alpha": alpha,
        "n_valid": int(len(valid)),
        "baseline_refusal_rate": float((valid["baseline_label"] == "refusal").mean()) if len(valid) else None,
        "intervened_refusal_rate": float((valid["intervened_label"] == "refusal").mean()) if len(valid) else None,
        "n_to_refusal": int(valid["baseline_to_refusal"].sum()),
        "n_to_compliance": int(valid["baseline_to_compliance"].sum()),
        "n_unchanged": int(valid["unchanged"].sum()),
    }
    summary_path = out_dir / f"{experiment_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    logger.info("Saved %s", result_path)
    logger.info("Summary: %s", summary)
    print(result_path)
    print(summary_path)
    print(summary)


if __name__ == "__main__":
    main()
