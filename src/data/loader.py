"""
Prompt loading and normalization.

Reads JSONL files and normalizes them into Prompt objects.
Supports:
  - harmful_prompts.jsonl  (label = "harmful")
  - benign_prompts.jsonl   (label = "benign")

Expected JSONL schema:
    {"prompt_id": str, "text": str, "category": str, "source": str}
    (label is inferred from which file is loaded)

Processed prompts are saved to data/processed/ as JSONL.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .schema import Prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level JSONL I/O
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
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, e)
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_prompts(
    path: str | Path,
    label: str,
    max_prompts: Optional[int] = None,
) -> list[Prompt]:
    """
    Load a JSONL file and return a list of Prompt objects.

    Args:
        path: Path to the JSONL file.
        label: "harmful" or "benign" — assigned to all loaded prompts.
        max_prompts: If set, truncate to this many prompts.

    Returns:
        List of Prompt objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    records = _read_jsonl(path)
    if max_prompts is not None:
        records = records[:max_prompts]

    prompts = []
    for rec in records:
        # Require at least prompt_id and text
        if "prompt_id" not in rec or "text" not in rec:
            logger.warning("Skipping record missing prompt_id or text: %s", rec)
            continue
        prompts.append(
            Prompt(
                prompt_id=rec["prompt_id"],
                text=rec["text"],
                label=label,
                category=rec.get("category", "unknown"),
                source=rec.get("source", "unknown"),
            )
        )

    logger.info("Loaded %d %s prompts from %s", len(prompts), label, path)
    return prompts


def load_harmful_prompts(
    path: str | Path,
    max_prompts: Optional[int] = None,
) -> list[Prompt]:
    """Load harmful prompts from JSONL."""
    return load_prompts(path, label="harmful", max_prompts=max_prompts)


def load_benign_prompts(
    path: str | Path,
    max_prompts: Optional[int] = None,
) -> list[Prompt]:
    """Load benign prompts from JSONL."""
    return load_prompts(path, label="benign", max_prompts=max_prompts)


# ---------------------------------------------------------------------------
# Saving processed prompt sets
# ---------------------------------------------------------------------------

def save_processed_prompts(prompts: list[Prompt], output_path: str | Path) -> None:
    """Save processed Prompt objects as JSONL."""
    records = [p.to_dict() for p in prompts]
    _write_jsonl(records, Path(output_path))


def load_processed_prompts(path: str | Path) -> list[Prompt]:
    """Load previously saved processed prompts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed prompt file not found: {path}")
    records = _read_jsonl(path)
    prompts = [Prompt.from_dict(r) for r in records]
    logger.info("Loaded %d processed prompts from %s", len(prompts), path)
    return prompts
