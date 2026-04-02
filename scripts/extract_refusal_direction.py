#!/usr/bin/env python3
"""
Extract the refusal direction via difference-in-means.

Runs the model on held-out harmful and benign prompts, collects residual-stream
activations at configured layers, and saves normalized direction vectors.

This must be run BEFORE run_tracing.py or run_patching.py.

Usage:
    python scripts/extract_refusal_direction.py
    python scripts/extract_refusal_direction.py --config configs/experiments/tracing.yaml
    python scripts/extract_refusal_direction.py --held-out-n 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, save_config_snapshot
from src.data.loader import load_harmful_prompts, load_benign_prompts
from src.generation.generator import load_model_and_tokenizer
from src.probing.direction import extract_refusal_direction, save_direction
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir, output_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract refusal direction via DiM.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--held-out-n", type=int, default=None,
                        help="Override held_out_n from config.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute even if direction file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_config:
        model_cfg = load_config(args.model_config)
        cfg.model = model_cfg.model
        if hasattr(model_cfg, "probing"):
            cfg.probing = model_cfg.probing

    if args.held_out_n is not None:
        cfg.probing.held_out_n = args.held_out_n

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "extract_direction.log",
    )

    import logging
    logger = logging.getLogger(__name__)

    direction_path = Path(cfg.probing.direction_save_path)

    if not args.no_resume and output_exists(direction_path):
        logger.info("Direction file already exists at %s. Use --no-resume to recompute.", direction_path)
        return

    logger.info("Extracting refusal direction for model: %s", cfg.model.name)

    max_p = getattr(cfg.data, "max_prompts", None)
    harmful = load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_p)
    benign = load_benign_prompts(cfg.data.benign_prompts, max_prompts=max_p)

    model, tokenizer = load_model_and_tokenizer(cfg)

    layers: list[int] = cfg.probing.layers
    held_out_n: int = getattr(cfg.probing, "held_out_n", 50)
    direction_position: int = getattr(cfg.probing, "direction_position", -1)

    directions = extract_refusal_direction(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful,
        benign_prompts=benign,
        layers=layers,
        direction_position=direction_position,
        held_out_n=held_out_n,
    )

    metadata = {
        "model_name": cfg.model.name,
        "layers": layers,
        "direction_position": direction_position,
        "held_out_n": held_out_n,
        "n_harmful": min(held_out_n, len(harmful)),
        "n_benign": min(held_out_n, len(benign)),
    }
    save_direction(directions, direction_path, metadata=metadata)

    # Save config snapshot alongside the direction
    ensure_dir(direction_path.parent)
    save_config_snapshot(cfg, direction_path.parent / "config_snapshot.yaml")
    logger.info("Done. Direction saved to %s", direction_path)


if __name__ == "__main__":
    main()
