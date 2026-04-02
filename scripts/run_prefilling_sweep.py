#!/usr/bin/env python3
"""
Prefilling sweep: vary k in {0, 1, 3, 5, 10} and measure refusal rate.

Outputs per-k JSONL files + a summary CSV with refusal rate vs k.

Usage:
    python scripts/run_prefilling_sweep.py
    python scripts/run_prefilling_sweep.py --config configs/experiments/prefilling_sweep.yaml
    python scripts/run_prefilling_sweep.py --k-values 0 1 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer, generate_responses
from src.classification.refusal_classifier import build_classifier_from_config, classify_responses
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefilling sweep over k values.")
    parser.add_argument("--config", type=str, default="configs/experiments/prefilling_sweep.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=None,
                        help="Override k values from config (e.g. --k-values 0 1 5).")
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

    if args.k_values is not None:
        cfg.prefilling.k_values = args.k_values

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "prefilling_sweep.log",
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting prefilling sweep. k_values = %s", cfg.prefilling.k_values)

    out_dir = ensure_dir(cfg.output.generations_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    max_p = getattr(cfg.data, "max_prompts", None)
    harmful = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_p)

    model, tokenizer = load_model_and_tokenizer(cfg)
    clf = build_classifier_from_config(cfg)

    all_results = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful,
        k_values=cfg.prefilling.k_values,
        cfg=cfg,
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    all_results = classify_responses(all_results, clf)
    save_jsonl(all_results, out_dir / "sweep_classified.jsonl")

    # Build summary table: refusal rate per k
    rows = []
    for k in cfg.prefilling.k_values:
        k_results = [r for r in all_results if r.get("prefix_k") == k]
        if not k_results:
            continue
        n_refusals = sum(1 for r in k_results if r.get("refusal_label") == "refusal")
        rows.append({
            "k": k,
            "n_prompts": len(k_results),
            "n_refusals": n_refusals,
            "refusal_rate": n_refusals / len(k_results),
        })
        logger.info("k=%2d: refusal_rate=%.2f (%d/%d)", k, rows[-1]["refusal_rate"], n_refusals, len(k_results))

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "refusal_rate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)

    print("\nRefusal Rate Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
