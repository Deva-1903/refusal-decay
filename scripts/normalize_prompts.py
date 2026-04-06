#!/usr/bin/env python3
"""
Normalize a user-provided JSONL or CSV file into the repo's Prompt schema.

Reads a source file where prompts may be stored under various field names
(e.g. "goal", "instruction", "prompt", "text") and writes a clean JSONL
file using the canonical schema:

    {"prompt_id": str, "text": str, "label": str, "category": str, "source": str}

Usage examples:

  # AdvBench CSV (has "goal" and "target" columns)
  python scripts/normalize_prompts.py \\
      --input data/raw/harmful_behaviors.csv \\
      --text-col goal \\
      --label harmful \\
      --source advbench \\
      --default-category harmful_behavior \\
      --output data/harmful_prompts.jsonl

  # Generic JSONL where the text field is "prompt"
  python scripts/normalize_prompts.py \\
      --input data/raw/my_prompts.jsonl \\
      --text-col prompt \\
      --label benign \\
      --source my_dataset \\
      --output data/benign_prompts.jsonl

  # Only take the first 500 rows
  python scripts/normalize_prompts.py \\
      --input data/raw/alpaca.jsonl \\
      --text-col instruction \\
      --label benign \\
      --source alpaca \\
      --max-prompts 500 \\
      --output data/benign_prompts.jsonl
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import Prompt
from src.utils.logging_utils import setup_logging
from src.utils.io_utils import ensure_dir

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d: %s", lineno, e)
    return records


def _read_csv(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    logger.info("Read %d rows from CSV %s", len(records), path)
    return records


def _read_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    elif suffix in {".jsonl", ".json", ".ndjson"}:
        return _read_jsonl(path)
    else:
        # Try JSONL first, fall back to CSV
        logger.warning(
            "Unrecognized extension '%s', trying JSONL then CSV.", suffix
        )
        try:
            return _read_jsonl(path)
        except Exception:
            return _read_csv(path)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_records(
    records: list[dict],
    text_col: str,
    label: str,
    source: str,
    default_category: str = "unknown",
    category_col: str = "",
    id_prefix: str = "",
    max_prompts: int = None,
) -> list[Prompt]:
    """
    Convert raw records to Prompt objects.

    Args:
        records: Raw dicts from CSV/JSONL.
        text_col: Field name containing the prompt text.
        label: "harmful" or "benign" — applied to all prompts.
        source: Dataset source name, e.g. "advbench".
        default_category: Category to use when category_col is empty or absent.
        category_col: Field name for category (empty string = use default_category).
        id_prefix: Prefix for auto-generated prompt_ids.
        max_prompts: Truncate to this many prompts if set.

    Returns:
        List of Prompt objects.
    """
    if max_prompts is not None:
        records = records[:max_prompts]

    prompts = []
    skipped = 0

    for i, rec in enumerate(records):
        # Extract text
        text = rec.get(text_col, "").strip()
        if not text:
            logger.warning("Row %d: empty text in column '%s', skipping.", i, text_col)
            skipped += 1
            continue

        # Extract or generate prompt_id
        if "prompt_id" in rec and rec["prompt_id"]:
            prompt_id = str(rec["prompt_id"])
        else:
            prefix = id_prefix or source
            prompt_id = f"{prefix}_{i:04d}"

        # Extract category
        category = default_category
        if category_col and category_col in rec and rec[category_col]:
            category = str(rec[category_col]).strip()

        prompts.append(Prompt(
            prompt_id=prompt_id,
            text=text,
            label=label,
            category=category,
            source=source,
        ))

    logger.info(
        "Normalized %d prompts (skipped %d with empty text).",
        len(prompts), skipped,
    )
    return prompts


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_prompts_jsonl(prompts: list[Prompt], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")
    logger.info("Wrote %d prompts to %s", len(prompts), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize external prompt files into the repo's Prompt schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input file (.jsonl or .csv).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Column/field name containing the prompt text (default: 'text').",
    )
    parser.add_argument(
        "--label", required=True, choices=["harmful", "benign"],
        help="Label to assign to all prompts ('harmful' or 'benign').",
    )
    parser.add_argument(
        "--source", required=True,
        help="Dataset source name, e.g. 'advbench' or 'alpaca'.",
    )
    parser.add_argument(
        "--category-col", default="",
        help="Column/field name for category (optional). If empty, uses --default-category.",
    )
    parser.add_argument(
        "--default-category", default="unknown",
        help="Category to use when no category column is present (default: 'unknown').",
    )
    parser.add_argument(
        "--id-prefix", default="",
        help="Prefix for auto-generated prompt_ids when no prompt_id field exists. "
             "Defaults to the value of --source.",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Truncate to this many prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Reading %s ...", input_path)
    records = _read_file(input_path)
    if not records:
        logger.error("No records read from %s", input_path)
        sys.exit(1)

    prompts = normalize_records(
        records=records,
        text_col=args.text_col,
        label=args.label,
        source=args.source,
        default_category=args.default_category,
        category_col=args.category_col,
        id_prefix=args.id_prefix,
        max_prompts=args.max_prompts,
    )

    if not prompts:
        logger.error("No valid prompts after normalization.")
        sys.exit(1)

    write_prompts_jsonl(prompts, Path(args.output))
    logger.info("Done.")


if __name__ == "__main__":
    main()
