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

    # Apply phrase-list refusal classification to baseline and patched outputs.
    # Field naming: baseline_phrase_label / patched_phrase_label to distinguish
    # from any future guard-based evaluation.
    for rec in results:
        if "error" in rec:
            rec["baseline_phrase_label"] = "error"
            rec["patched_phrase_label"] = "error"
            rec["refusal_restored"] = None   # compliance → refusal (expected positive effect)
            rec["refusal_lost"] = None       # refusal → compliance (unexpected direction)
            continue
        bl = clf.classify(rec.get("baseline_text", ""))
        pl = clf.classify(rec.get("patched_text", ""))
        rec["baseline_phrase_label"] = bl
        rec["patched_phrase_label"] = pl
        # Primary hypothesis: patching restores refusal where baseline was compliant.
        # compliance → refusal = positive causal effect (refusal signal mattered)
        rec["refusal_restored"] = (bl == "compliance" and pl == "refusal")
        # Secondary: did patching accidentally destroy a refusal?
        # refusal → compliance = unexpected direction, would undermine hypothesis
        rec["refusal_lost"] = (bl == "refusal" and pl == "compliance")

    # Save full classified results
    from src.utils.io_utils import save_jsonl
    save_jsonl(results, out_dir / "patching_classified.jsonl")

    # Summary table — print breakdown by expected vs unexpected effect direction.
    valid = [r for r in results if "error" not in r]
    if valid:
        n_total = len(valid)
        n_restored = sum(1 for r in valid if r.get("refusal_restored"))
        n_lost = sum(1 for r in valid if r.get("refusal_lost"))
        n_no_change = n_total - n_restored - n_lost
        logger.info(
            "Patching summary (n=%d): refusal_restored=%d (%.1f%%), "
            "refusal_lost=%d (%.1f%%), no_change=%d (%.1f%%)",
            n_total,
            n_restored, 100 * n_restored / n_total,
            n_lost, 100 * n_lost / n_total,
            n_no_change, 100 * n_no_change / n_total,
        )
        print(f"\nPatching results (n={n_total}):")
        print(f"  compliance → refusal (expected direction): {n_restored}/{n_total} ({100*n_restored/n_total:.1f}%)")
        print(f"  refusal → compliance (unexpected):         {n_lost}/{n_total} ({100*n_lost/n_total:.1f}%)")
        print(f"  no change:                                 {n_no_change}/{n_total} ({100*n_no_change/n_total:.1f}%)")

    logger.info("Patching experiment complete.")


if __name__ == "__main__":
    main()
