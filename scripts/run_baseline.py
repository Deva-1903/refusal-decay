#!/usr/bin/env python3
"""
Baseline generation: run the model on all harmful prompts with no prefilling (k=0).

Outputs are saved to outputs/generations/baseline/ as JSONL.
A refusal classifier is applied to each response.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --config configs/experiments/baseline.yaml
    python scripts/run_baseline.py --model-config configs/model_8b.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts, load_benign_prompts, save_processed_prompts
from src.generation.generator import load_model_and_tokenizer, generate_responses
from src.classification.refusal_classifier import build_classifier_from_config, classify_responses
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline generation (k=0).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/baseline.yaml",
        help="Experiment config path (merged on top of configs/base.yaml).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional model override config (e.g. configs/model_8b.yaml).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun even if outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    # Apply model override if specified
    if args.model_config:
        from src.config import load_config as _lc
        model_cfg = _lc(args.model_config)
        # Merge model section only
        cfg.model = model_cfg.model
        if hasattr(model_cfg, "probing"):
            cfg.probing = model_cfg.probing

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "baseline.log",
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting baseline generation.")
    logger.info("Model: %s", cfg.model.name)

    # Save config snapshot
    out_dir = ensure_dir(cfg.output.generations_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    # Load prompts
    max_p = getattr(cfg.data, "max_prompts", None)
    harmful = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_p)
    benign = load_benign_prompts(cfg.data.benign_prompts, max_prompts=max_p)

    # Save processed prompts for reference
    processed_dir = ensure_dir(cfg.data.processed_dir)
    save_processed_prompts(harmful, processed_dir / "harmful_prompts.jsonl")
    save_processed_prompts(benign, processed_dir / "benign_prompts.jsonl")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Generate (k=0 only for baseline)
    results = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful,
        k_values=[0],
        cfg=cfg,
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    # Classify
    clf = build_classifier_from_config(cfg)
    results = classify_responses(results, clf)

    # Save classified results
    save_jsonl(results, out_dir / "baseline_classified.jsonl")

    # Print summary
    n_refusals = sum(1 for r in results if r.get("refusal_phrase_label") == "refusal")
    logger.info(
        "Baseline complete. Refusal rate: %d/%d = %.1f%%",
        n_refusals, len(results), 100 * n_refusals / max(len(results), 1),
    )


if __name__ == "__main__":
    main()
