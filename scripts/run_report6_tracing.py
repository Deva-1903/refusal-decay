#!/usr/bin/env python3
"""
Run the Report 6 observational tracing block.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, save_config_snapshot
from src.data.loader import load_benign_prompts, load_harmful_prompts
from src.generation.generator import load_model_and_tokenizer
from src.probing.direction import load_direction
from src.probing.tracing import trace_projections
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Report 6 tracing.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/report6/trace_report6.yaml",
        help="Tracing config.",
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


def _load_trace_prompts(cfg, dataset: str):
    max_prompts = getattr(cfg.data, "max_prompts", None)
    if dataset == "harmful":
        return load_harmful_prompts(cfg.data.harmful_prompts, max_prompts=max_prompts)
    if dataset == "benign":
        return load_benign_prompts(cfg.data.benign_prompts, max_prompts=max_prompts)
    raise ValueError(f"Unsupported tracing dataset '{dataset}'.")


def main() -> None:
    args = parse_args()
    cfg = _apply_model_override(load_config(args.config), args.model_config)

    if args.direction_path:
        cfg.probing.direction_save_path = args.direction_path

    setup_logging(
        level=getattr(cfg.logging, "level", "INFO"),
        log_file=Path(cfg.output.log_dir) / "report6_tracing.log",
    )

    import logging

    logger = logging.getLogger(__name__)

    dataset = getattr(getattr(cfg, "report6", object()), "trace_dataset", "harmful")
    logger.info("Starting Report 6 tracing.")
    logger.info("Model: %s", cfg.model.name)
    logger.info("Trace dataset: %s", dataset)
    logger.info("Trace k values: %s", cfg.prefilling.k_values)
    logger.info("Trace layers: %s", cfg.probing.layers)

    out_dir = ensure_dir(cfg.tracing.output_dir)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    prompts = _load_trace_prompts(cfg, dataset=dataset)
    directions = load_direction(cfg.probing.direction_save_path)
    model, tokenizer = load_model_and_tokenizer(cfg)

    df = trace_projections(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        k_values=cfg.prefilling.k_values,
        directions=directions,
        cfg=cfg,
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    if not df.empty:
        df["condition_name"] = df.apply(
            lambda row: f"{row['label']}_k{int(row['prefix_k']):02d}",
            axis=1,
        )
        combined_path = out_dir / f"traces_all.{cfg.tracing.save_format}"
        if cfg.tracing.save_format == "parquet":
            df.to_parquet(combined_path, index=False)
        else:
            df.to_csv(combined_path, index=False)
        logger.info("Saved combined traces to %s (%d rows)", combined_path, len(df))
    else:
        logger.warning("Tracing returned no rows.")


if __name__ == "__main__":
    main()
