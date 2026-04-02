#!/usr/bin/env python3
"""
Pilot patching experiment: inject refusal-direction component from source token
position ts into target position tt and observe generation change.

Requires: run extract_refusal_direction.py first.

Usage:
    python scripts/run_patching.py
    python scripts/run_patching.py --config configs/experiments/patching.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer
from src.probing.direction import load_direction
from src.patching.patch import run_patching_experiment
from src.classification.refusal_classifier import build_classifier_from_config
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot patching experiment.")
    parser.add_argument("--config", type=str, default="configs/experiments/patching.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--direction-path", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_config:
        model_cfg = load_config(args.model_config)
        cfg.model = model_cfg.model
        if hasattr(model_cfg, "probing"):
            cfg.probing = model_cfg.probing

    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "patching.log",
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting patching experiment.")

    out_dir = ensure_dir(cfg.patching.output_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    max_p = getattr(cfg.data, "max_prompts", None)
    harmful = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_p)

    directions = load_direction(cfg.probing.direction_save_path)
    model, tokenizer = load_model_and_tokenizer(cfg)
    clf = build_classifier_from_config(cfg)

    results = run_patching_experiment(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful,
        directions=directions,
        cfg=cfg,
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    # Apply refusal classification to baseline and patched outputs
    for rec in results:
        if "error" in rec:
            continue
        rec["baseline_label"] = clf.classify(rec.get("baseline_text", ""))
        rec["patched_label"] = clf.classify(rec.get("patched_text", ""))
        rec["label_changed"] = rec["baseline_label"] != rec["patched_label"]

    # Save full classified results
    from src.utils.io_utils import save_jsonl
    save_jsonl(results, out_dir / "patching_classified.jsonl")

    # Summary: how often did patching change the refusal label?
    valid = [r for r in results if "error" not in r]
    if valid:
        n_changed = sum(1 for r in valid if r.get("label_changed"))
        logger.info(
            "Label changed in %d/%d cases (%.1f%%).",
            n_changed, len(valid), 100 * n_changed / len(valid),
        )
        print(f"\nPatching changed refusal label: {n_changed}/{len(valid)} ({100*n_changed/len(valid):.1f}%)")

    logger.info("Patching experiment complete.")


if __name__ == "__main__":
    main()
