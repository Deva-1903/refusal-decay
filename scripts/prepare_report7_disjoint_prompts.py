#!/usr/bin/env python3
"""
Build disjoint prompt sets for Report 7.

Report 6 used the same 50 harmful + 50 benign prompts for direction extraction
that overlapped with the 25 traced prompts. Report 7 fixes this by holding the
direction-extraction prompts disjoint from the traced/intervention prompts.

Splits:
  - Traced/intervention set (used by tracing, patching, additive intervention):
      first 25 of each master JSONL  -> data/processed/report7_traced_*.jsonl
  - Held-out direction-extraction set (used by extract_refusal_direction):
      next 50 of each master JSONL   -> data/processed/report7_direction_*.jsonl

The traced set is identical to what Report 6 used (first 25), so behavioral and
tracing results stay comparable. Only the direction-extraction prompts change.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HARMFUL = ROOT / "data" / "harmful_prompts.jsonl"
DEFAULT_BENIGN = ROOT / "data" / "benign_prompts.jsonl"
DEFAULT_OUT = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disjoint prompt sets for Report 7.")
    parser.add_argument("--harmful", type=str, default=str(DEFAULT_HARMFUL))
    parser.add_argument("--benign", type=str, default=str(DEFAULT_BENIGN))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--traced-n", type=int, default=25,
                        help="Number of prompts in the traced/intervention set (must match the rest of R7).")
    parser.add_argument("--direction-n", type=int, default=50,
                        help="Number of prompts in the held-out direction-extraction set.")
    parser.add_argument("--prefix", type=str, default="report7",
                        help="Output filename prefix, e.g. 'report7' or 'paper'.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _split_and_write(src: Path, out_dir: Path, name: str, traced_n: int, direction_n: int, prefix: str = "report7") -> dict:
    records = _read_jsonl(src)
    if len(records) < traced_n + direction_n:
        raise ValueError(
            f"{src} has only {len(records)} prompts; need at least {traced_n + direction_n} "
            f"({traced_n} traced + {direction_n} direction)."
        )
    traced = records[:traced_n]
    direction = records[traced_n : traced_n + direction_n]

    traced_path = out_dir / f"{prefix}_traced_{name}.jsonl"
    direction_path = out_dir / f"{prefix}_direction_{name}.jsonl"
    _write_jsonl(traced, traced_path)
    _write_jsonl(direction, direction_path)

    traced_ids = {record.get("prompt_id") for record in traced}
    direction_ids = {record.get("prompt_id") for record in direction}
    overlap = traced_ids & direction_ids
    if overlap:
        raise RuntimeError(f"Disjoint check failed for {name}: overlapping prompt_ids {overlap}")

    return {
        "name": name,
        "traced_n": len(traced),
        "direction_n": len(direction),
        "traced_path": str(traced_path),
        "direction_path": str(direction_path),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    summaries = [
        _split_and_write(Path(args.harmful), out_dir, "harmful", args.traced_n, args.direction_n, args.prefix),
        _split_and_write(Path(args.benign), out_dir, "benign", args.traced_n, args.direction_n, args.prefix),
    ]
    for entry in summaries:
        print(
            f"{entry['name']}: traced={entry['traced_n']} -> {entry['traced_path']} | "
            f"direction={entry['direction_n']} -> {entry['direction_path']}"
        )


if __name__ == "__main__":
    main()
