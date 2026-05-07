#!/usr/bin/env python3
"""
Report 7 cross-condition patching.

Source: baseline harmful k=0 input/prefill state.
Target: attacked harmful k=3 generation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.classification.refusal_classifier import build_classifier_from_config
from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer
from src.patching.report7_interventions import (
    extract_direction_component,
    generate_baseline,
    generate_with_component_patch,
)
from src.probing.direction import load_direction
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Report 7 cross-condition patching.")
    parser.add_argument("--config", type=str, default="configs/experiments/report7/patching_report7.yaml")
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
    text = text or ""
    return text[:limit]


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    valid = df[df["error"].fillna("") == ""].copy()
    if valid.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    summary = (
        valid.groupby(["source_condition", "target_condition", "layer", "source_position", "target_position", "mode"])
        .agg(
            n_prompts=("prompt_id", "size"),
            restored=("restored_refusal", "sum"),
            lost=("lost_refusal", "sum"),
            unchanged=("unchanged", "sum"),
            mean_source_projection=("source_projection", "mean"),
            mean_target_projection_before=("target_projection_before", "mean"),
            mean_target_projection_after=("target_projection_after", "mean"),
        )
        .reset_index()
    )
    summary["restoration_rate"] = summary["restored"] / summary["n_prompts"]
    summary["loss_rate"] = summary["lost"] / summary["n_prompts"]
    summary.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)
    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path
    if args.max_prompts is not None:
        cfg.data.max_prompts = args.max_prompts

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "report7_cross_condition_patching.log",
    )

    import logging

    logger = logging.getLogger(__name__)
    out_dir = ensure_dir(cfg.patching.output_dir)
    save_config_snapshot(cfg, out_dir / "cross_condition_config_snapshot.yaml")

    source_condition = getattr(cfg.report7, "source_condition", "harmful_k00")
    target_condition = getattr(cfg.report7, "target_condition", "harmful_k03")
    source_k = int(getattr(cfg.prefilling, "source_k", 0))
    target_k = int(getattr(cfg.prefilling, "target_k", 3))
    prefix_text = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    max_new_tokens = int(getattr(cfg.generation, "max_new_tokens", 64))
    source_position = int(cfg.patching.source_position)
    layers = [int(layer) for layer in cfg.patching.layers]
    target_positions = [int(position) for position in cfg.patching.target_positions]
    mode = getattr(cfg.patching, "mode", "replace")

    logger.info("Source condition: %s", source_condition)
    logger.info("Target condition: %s", target_condition)
    logger.info("Layers: %s", layers)
    logger.info("Target positions: %s", target_positions)

    prompts = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=getattr(cfg.data, "max_prompts", None))
    ids = [prompt.prompt_id for prompt in prompts]
    if any(not prompt_id for prompt_id in ids):
        logger.warning("Some prompts have missing prompt_id; results will still include the loaded Prompt identifier.")
    if len(ids) != len(set(ids)):
        logger.warning("Duplicate prompt_id values detected; stable prompt matching may be ambiguous.")

    directions = load_direction(cfg.probing.direction_save_path)
    model, tokenizer = load_model_and_tokenizer(cfg)
    classifier = build_classifier_from_config(cfg)

    baseline_cache: dict[str, tuple[str, list[int], str]] = {}
    source_cache: dict[tuple[str, int], tuple[float, object]] = {}
    rows: list[dict] = []

    for layer in layers:
        if layer not in directions:
            logger.warning("No refusal direction for layer %d; skipping.", layer)
            continue
        direction = directions[layer]
        for target_position in target_positions:
            if target_position < 0:
                logger.warning("Skipping unsupported target_position=%d; generated-token positions must be >= 0.", target_position)
                continue
            logger.info("Running layer=%d target_position=%d over %d prompts", layer, target_position, len(prompts))
            for prompt in prompts:
                base = {
                    "prompt_id": prompt.prompt_id,
                    "source_condition": source_condition,
                    "target_condition": target_condition,
                    "layer": layer,
                    "source_position": source_position,
                    "target_position": target_position,
                    "mode": mode,
                    "baseline_attacked_label": None,
                    "patched_label": None,
                    "restored_refusal": False,
                    "lost_refusal": False,
                    "unchanged": False,
                    "source_projection": None,
                    "target_projection_before": None,
                    "target_projection_after": None,
                    "baseline_attacked_output_truncated": "",
                    "patched_output_truncated": "",
                    "error": "",
                }
                try:
                    if prompt.prompt_id not in baseline_cache:
                        baseline_text, baseline_ids = generate_baseline(
                            model, tokenizer, prompt, target_k, prefix_text, max_new_tokens
                        )
                        baseline_cache[prompt.prompt_id] = (
                            baseline_text,
                            baseline_ids,
                            classifier.classify(baseline_text),
                        )
                    baseline_text, _, baseline_label = baseline_cache[prompt.prompt_id]

                    source_key = (prompt.prompt_id, layer)
                    if source_key not in source_cache:
                        source_cache[source_key] = extract_direction_component(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            k=source_k,
                            prefix_text=prefix_text,
                            layer=layer,
                            position=source_position,
                            direction=direction,
                        )
                    source_projection, source_component = source_cache[source_key]

                    patched = generate_with_component_patch(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        target_k=target_k,
                        prefix_text=prefix_text,
                        layer=layer,
                        target_position=target_position,
                        direction=direction,
                        source_component=source_component,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                    )
                    patched_label = classifier.classify(patched.text)
                    restored = baseline_label == "compliance" and patched_label == "refusal"
                    lost = baseline_label == "refusal" and patched_label == "compliance"
                    base.update(
                        baseline_attacked_label=baseline_label,
                        patched_label=patched_label,
                        restored_refusal=restored,
                        lost_refusal=lost,
                        unchanged=not restored and not lost,
                        source_projection=source_projection,
                        target_projection_before=patched.projection_before,
                        target_projection_after=patched.projection_after,
                        baseline_attacked_output_truncated=_truncate(baseline_text),
                        patched_output_truncated=_truncate(patched.text),
                    )
                except Exception as exc:
                    logger.error("Error on prompt=%s layer=%d target=%d: %s", prompt.prompt_id, layer, target_position, exc)
                    base["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(base)

    df = pd.DataFrame(rows)
    result_path = out_dir / "cross_condition_patching_results.csv"
    summary_path = out_dir / "cross_condition_patching_summary.csv"
    df.to_csv(result_path, index=False)
    _write_summary(df, summary_path)
    logger.info("Saved cross-condition patching results to %s", result_path)
    logger.info("Saved cross-condition patching summary to %s", summary_path)
    print(result_path)
    print(summary_path)


if __name__ == "__main__":
    main()
