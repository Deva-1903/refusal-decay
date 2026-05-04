#!/usr/bin/env python3
"""
Run the Report 6 generation block across the configured conditions.

Default conditions:
  - harmful_k00
  - harmful_k03
  - harmful_k10 (optional but included when configured)
  - benign_k00
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.classification.refusal_classifier import build_classifier_from_config, classify_responses
from src.config import load_config, save_config_snapshot
from src.data.loader import load_benign_prompts, load_harmful_prompts
from src.generation.generator import generate_responses, load_model_and_tokenizer
from src.utils.io_utils import ensure_dir, load_jsonl, output_exists, save_jsonl
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Report 6 generation conditions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/report6/generation_report6.yaml",
        help="Report 6 generation config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional model override config.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of configured condition names to run.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun even if classified condition outputs already exist.",
    )
    return parser.parse_args()


def _apply_model_override(cfg, model_config_path: str | None):
    if not model_config_path:
        return cfg
    model_cfg = load_config(model_config_path)
    cfg.model = model_cfg.model
    if hasattr(model_cfg, "probing"):
        cfg.probing = model_cfg.probing
    return cfg


def _load_prompt_set(cfg, dataset: str, max_prompts: int | None):
    if dataset == "harmful":
        return load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_prompts)
    if dataset == "benign":
        return load_benign_prompts(cfg.data.benign_prompts, max_prompts=max_prompts)
    raise ValueError(f"Unsupported Report 6 dataset '{dataset}'. Expected 'harmful' or 'benign'.")


def _condition_summary(records: list[dict], condition_name: str, dataset: str, k: int) -> dict:
    n_total = len(records)
    n_refusal = sum(1 for rec in records if rec.get("refusal_phrase_label") == "refusal")
    n_compliance = sum(1 for rec in records if rec.get("refusal_phrase_label") == "compliance")
    n_error = sum(1 for rec in records if rec.get("refusal_phrase_label") == "error")
    n_valid = n_refusal + n_compliance
    refusal_rate = (n_refusal / n_valid) if n_valid else None
    return {
        "condition_name": condition_name,
        "dataset_name": dataset,
        "prefix_k": k,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_refusal": n_refusal,
        "n_compliance": n_compliance,
        "n_error": n_error,
        "refusal_rate": refusal_rate,
    }


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)

    log_dir = ensure_dir(cfg.output.log_dir)
    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=log_dir / "report6_generation.log",
    )

    import logging

    logger = logging.getLogger(__name__)

    conditions = list(getattr(cfg.report6, "generation_conditions", []))
    if not conditions:
        raise ValueError("No report6.generation_conditions found in the config.")

    requested = set(args.conditions or [])
    if requested:
        known = {condition["name"] for condition in conditions}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
        conditions = [condition for condition in conditions if condition["name"] in requested]

    logger.info("Starting Report 6 generation.")
    logger.info("Model: %s", cfg.model.name)
    logger.info("Conditions: %s", [condition["name"] for condition in conditions])
    logger.info("Generation max_new_tokens: %s", getattr(cfg.generation, "max_new_tokens", None))

    generations_dir = ensure_dir(cfg.output.generations_dir)
    ensure_dir(getattr(cfg.report6, "summaries_dir", generations_dir.parent / "summaries"))
    save_config_snapshot(cfg, generations_dir / "config_snapshot.yaml")

    model, tokenizer = load_model_and_tokenizer(cfg)
    classifier = build_classifier_from_config(cfg)

    all_records: list[dict] = []
    summary_rows: list[dict] = []

    for condition in conditions:
        condition_name = condition["name"]
        dataset = condition["dataset"]
        k = int(condition["k"])
        max_prompts = condition.get("max_prompts", getattr(cfg.data, "max_prompts", None))

        condition_dir = ensure_dir(generations_dir / condition_name)
        save_config_snapshot(cfg, condition_dir / "config_snapshot.yaml")
        classified_path = condition_dir / "classified.jsonl"

        logger.info(
            "Condition %s: dataset=%s k=%d max_prompts=%s output_dir=%s",
            condition_name,
            dataset,
            k,
            max_prompts,
            condition_dir,
        )

        if not args.no_resume and output_exists(classified_path):
            logger.info("Loading cached classified outputs for %s from %s", condition_name, classified_path)
            condition_records = load_jsonl(classified_path)
        else:
            prompts = _load_prompt_set(cfg, dataset=dataset, max_prompts=max_prompts)
            condition_records = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                k_values=[k],
                cfg=cfg,
                output_dir=condition_dir,
                resume=not args.no_resume,
            )
            condition_records = classify_responses(condition_records, classifier)

        for record in condition_records:
            record["condition_name"] = condition_name
            record["dataset_name"] = dataset

        save_jsonl(condition_records, classified_path)
        all_records.extend(condition_records)
        summary_rows.append(_condition_summary(condition_records, condition_name, dataset, k))

    combined_path = generations_dir / "report6_generation_combined.jsonl"
    save_jsonl(all_records, combined_path)

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset_name", "prefix_k", "condition_name"])
    summary_path = generations_dir / "report6_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info("Saved combined generation outputs to %s", combined_path)
    logger.info("Saved generation summary to %s", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
