#!/usr/bin/env python3
"""
Run the Report 6 targeted patching block.
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
from src.patching.patch import run_patching_experiment
from src.probing.direction import load_direction
from src.utils.io_utils import ensure_dir, load_jsonl, output_exists, save_jsonl
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Report 6 patching.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/report6/patch_report6.yaml",
        help="Patching config.",
    )
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--direction-path", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _apply_model_override(cfg, model_config_path: str | None):
    if not model_config_path:
        return cfg
    model_cfg = load_config(model_config_path)
    cfg.model = model_cfg.model
    if hasattr(model_cfg, "probing"):
        cfg.probing = model_cfg.probing
    return cfg


def _summarize_patch_results(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([record for record in records if "error" not in record])
    if df.empty:
        return df

    summary = (
        df.groupby(["condition_name", "layer", "source_position", "target_position", "mode"])
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
    return summary


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)

    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "report6_patching.log",
    )

    import logging

    logger = logging.getLogger(__name__)

    out_dir = ensure_dir(cfg.patching.output_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")
    classified_path = out_dir / "patching_classified.jsonl"

    attack_k = int(cfg.prefilling.k_values[0])
    logger.info("Starting Report 6 patching.")
    logger.info("Model: %s", cfg.model.name)
    logger.info("Attack condition: harmful_k%02d", attack_k)
    logger.info("Layers: %s", cfg.patching.layers)
    logger.info("Target positions: %s", cfg.patching.target_positions)
    logger.info("Source position: %s", cfg.patching.source_position)

    if not args.no_resume and output_exists(classified_path):
        logger.info("Loading cached classified patch outputs from %s", classified_path)
        results = load_jsonl(classified_path)
    else:
        harmful = load_harmful_prompts(
            cfg.data.harmful_prompts,
            max_prompts=getattr(cfg.data, "max_prompts", None),
        )
        directions = load_direction(cfg.probing.direction_save_path)
        model, tokenizer = load_model_and_tokenizer(cfg)
        classifier = build_classifier_from_config(cfg)

        results = run_patching_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful,
            directions=directions,
            cfg=cfg,
            output_dir=out_dir,
            resume=not args.no_resume,
        )

        for record in results:
            record["condition_name"] = f"harmful_k{attack_k:02d}"
            record["dataset_name"] = "harmful"
            if "error" in record:
                record["baseline_phrase_label"] = "error"
                record["patched_phrase_label"] = "error"
                record["refusal_restored"] = None
                record["refusal_lost"] = None
                continue
            baseline_label = classifier.classify(record.get("baseline_text", ""))
            patched_label = classifier.classify(record.get("patched_text", ""))
            record["baseline_phrase_label"] = baseline_label
            record["patched_phrase_label"] = patched_label
            record["refusal_restored"] = (
                baseline_label == "compliance" and patched_label == "refusal"
            )
            record["refusal_lost"] = (
                baseline_label == "refusal" and patched_label == "compliance"
            )

        save_jsonl(results, classified_path)

    for record in results:
        record.setdefault("condition_name", f"harmful_k{attack_k:02d}")
        record.setdefault("dataset_name", "harmful")

    summary_df = _summarize_patch_results(results)
    summary_path = out_dir / "patching_summary.csv"
    if summary_df.empty:
        logger.warning("No valid patching records were available for summary.")
    else:
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved patching summary to %s", summary_path)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
