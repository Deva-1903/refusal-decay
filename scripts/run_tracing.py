#!/usr/bin/env python3
"""
Tracing: collect residual-stream projections onto the refusal direction
at each token position during generation, for each k value.

Requires: run extract_refusal_direction.py first.

Outputs: Parquet/CSV files with columns:
    prompt_id, label, category, prefix_k, step, layer, projection

Usage:
    python scripts/run_tracing.py
    python scripts/run_tracing.py --config configs/experiments/tracing.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer
from src.probing.direction import load_direction
from src.probing.tracing import trace_projections
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace refusal-direction projections.")
    parser.add_argument("--config", type=str, default="configs/experiments/tracing.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--direction-path", type=str, default=None,
                        help="Override path to saved direction file.")
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
        log_file=Path(cfg.output.log_dir) / "tracing.log",
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting projection tracing.")

    out_dir = ensure_dir(cfg.tracing.output_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    max_p = getattr(cfg.data, "max_prompts", None)
    harmful = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_p)

    directions = load_direction(cfg.probing.direction_save_path)

    model, tokenizer = load_model_and_tokenizer(cfg)

    df = trace_projections(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful,
        k_values=cfg.prefilling.k_values,
        directions=directions,
        cfg=cfg,
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    if not df.empty:
        combined_path = out_dir / f"traces_all.{cfg.tracing.save_format}"
        if cfg.tracing.save_format == "parquet":
            df.to_parquet(combined_path, index=False)
        else:
            df.to_csv(combined_path, index=False)
        logger.info("Combined trace saved: %s (%d rows)", combined_path, len(df))

    logger.info("Tracing complete.")


if __name__ == "__main__":
    main()
