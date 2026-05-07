#!/usr/bin/env python3
"""
Run or verify the Report 7 behavioral generation conditions.

This is intentionally close to the Report 6 generation runner, with one extra
guard: cached all-error condition outputs are never reused silently.
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
    parser = argparse.ArgumentParser(description="Run Report 7 generation verification.")
    parser.add_argument("--config", type=str, default="configs/experiments/report7/generation_report7.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--no-reuse-report6",
        action="store_true",
        help="Do not copy valid Report 6 generation caches into outputs/report7.",
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
    raise ValueError(f"Unsupported dataset: {dataset}")


def _is_all_error(records: list[dict]) -> bool:
    return bool(records) and all(record.get("error") for record in records)


def _load_report6_condition(cfg, condition_name: str) -> list[dict] | None:
    report6_root = Path(getattr(cfg.report7, "report6_root", "outputs/report6"))
    path = report6_root / "generations" / condition_name / "classified.jsonl"
    if not path.exists():
        return None
    records = load_jsonl(path)
    if _is_all_error(records):
        return None
    return records


def _condition_summary(records: list[dict], condition_name: str, dataset: str, k: int) -> dict:
    n_total = len(records)
    n_error = sum(1 for record in records if record.get("refusal_phrase_label") == "error" or record.get("error"))
    n_refusal = sum(1 for record in records if record.get("refusal_phrase_label") == "refusal")
    n_compliance = sum(1 for record in records if record.get("refusal_phrase_label") == "compliance")
    n_valid = n_refusal + n_compliance
    return {
        "condition_name": condition_name,
        "dataset_name": dataset,
        "prefix_k": k,
        "n_total": n_total,
        "n_refusal": n_refusal,
        "n_compliance": n_compliance,
        "n_error": n_error,
        "n_valid": n_valid,
        "refusal_rate": n_refusal / n_valid if n_valid else None,
    }


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)

    log_dir = ensure_dir(cfg.output.log_dir)
    setup_logging(level=getattr(cfg.logging, "level", "INFO"), log_file=log_dir / "report7_generation.log")

    import logging

    logger = logging.getLogger(__name__)
    conditions = list(getattr(cfg.report7, "generation_conditions", []))
    if args.conditions:
        requested = set(args.conditions)
        known = {condition["name"] for condition in conditions}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
        conditions = [condition for condition in conditions if condition["name"] in requested]

    logger.info("Starting Report 7 generation verification.")
    logger.info("Model: %s", cfg.model.name)
    logger.info("Conditions: %s", [condition["name"] for condition in conditions])

    generations_dir = ensure_dir(cfg.output.generations_dir)
    save_config_snapshot(cfg, generations_dir / "config_snapshot.yaml")

    model = tokenizer = classifier = None
    all_records: list[dict] = []
    rows: list[dict] = []

    for condition in conditions:
        condition_name = condition["name"]
        dataset = condition["dataset"]
        k = int(condition["k"])
        max_prompts = condition.get("max_prompts", getattr(cfg.data, "max_prompts", None))
        condition_dir = ensure_dir(generations_dir / condition_name)
        classified_path = condition_dir / "classified.jsonl"

        condition_records: list[dict] | None = None
        if not args.no_resume and output_exists(classified_path):
            cached = load_jsonl(classified_path)
            if _is_all_error(cached):
                logger.warning("Cached %s is all errors; recomputing instead of reusing it.", classified_path)
            else:
                logger.info("Loading cached classified outputs for %s", condition_name)
                condition_records = cached

        if condition_records is None:
            if not args.no_reuse_report6 and not args.no_resume:
                report6_records = _load_report6_condition(cfg, condition_name)
                if report6_records is not None:
                    logger.info("Reusing valid Report 6 outputs for %s", condition_name)
                    condition_records = report6_records

        if condition_records is None:
            if model is None or tokenizer is None or classifier is None:
                model, tokenizer = load_model_and_tokenizer(cfg)
                classifier = build_classifier_from_config(cfg)
            prompts = _load_prompt_set(cfg, dataset=dataset, max_prompts=max_prompts)
            condition_records = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                k_values=[k],
                cfg=cfg,
                output_dir=condition_dir,
                resume=False,
            )
            condition_records = classify_responses(condition_records, classifier)

        for record in condition_records:
            record["condition_name"] = condition_name
            record["dataset_name"] = dataset
        save_jsonl(condition_records, classified_path)
        all_records.extend(condition_records)
        rows.append(_condition_summary(condition_records, condition_name, dataset, k))

    combined_path = generations_dir / "report7_generation_combined.jsonl"
    save_jsonl(all_records, combined_path)
    summary = pd.DataFrame(rows).sort_values(["dataset_name", "prefix_k", "condition_name"])
    summary_path = generations_dir / "report7_generation_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary_dir = ensure_dir(getattr(cfg.report7, "summaries_dir", "outputs/report7/summaries"))
    summary.to_csv(summary_dir / "generation_refusal_rates_by_condition.csv", index=False)
    labels_df = pd.DataFrame(all_records)
    if not labels_df.empty:
        label_cols = [
            "prompt_id",
            "condition_name",
            "dataset_name",
            "prefix_k",
            "refusal_phrase_label",
            "matched_phrase",
        ]
        available_cols = [col for col in label_cols if col in labels_df.columns]
        labels_df[available_cols].to_csv(summary_dir / "generation_prompt_labels.csv", index=False)

    logger.info("Saved combined generation outputs to %s", combined_path)
    logger.info("Saved generation summary to %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
