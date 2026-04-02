"""
File I/O helpers used across the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug("Saved JSON to %s", path)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    logger.debug("Saved %d records to %s", len(records), path)


def load_jsonl(path: Union[str, Path]) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def output_exists(path: Union[str, Path]) -> bool:
    """Check whether a file output already exists (for resume-safe runs)."""
    return Path(path).exists()
