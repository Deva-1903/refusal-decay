#!/usr/bin/env python3
"""
Manual spot-check helper for the phrase-classifier validity threat.

Report 6 Section 7.3 flagged the phrase classifier as the top threat to
internal validity. This script supports the cheapest meaningful fix:
manually re-label a stratified sample of outputs and report how often
the auto-classifier disagrees with the human label.

Two-phase workflow:

  Phase 1 (generate template):
    python scripts/run_classifier_spot_check.py generate

  --> Writes outputs/report7/classifier_spot_check/spot_check.yaml with
      ~20 stratified items. Each item has auto_label set and your_label
      placeholder. Open the file and replace each `your_label: TODO`
      with `refusal`, `compliance`, or `ambiguous`.

  Phase 2 (score after you fill it in):
    python scripts/run_classifier_spot_check.py score

  --> Writes outputs/report7/classifier_spot_check/spot_check_report.md
      with: per-condition agreement rates, flip rate vs auto, and the
      borderline items. Also writes a CSV for tables in the report.

The two-phase split is deliberate: scoring is a no-op if you haven't
edited the YAML yet, and you can run it again as you refine labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.utils.io_utils import ensure_dir


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATIONS = ROOT / "outputs" / "report7" / "generations"
DEFAULT_OUT_DIR = ROOT / "outputs" / "report7" / "classifier_spot_check"
DEFAULT_TEMPLATE = DEFAULT_OUT_DIR / "spot_check.yaml"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "spot_check_report.md"
DEFAULT_CSV = DEFAULT_OUT_DIR / "spot_check_results.csv"

VALID_HUMAN_LABELS = {"refusal", "compliance", "ambiguous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["generate", "score"], help="generate template or score it")
    parser.add_argument("--generations-dir", type=str, default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--template", type=str, default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument(
        "--per-cell",
        type=int,
        default=5,
        help="Items per (condition, auto_label) cell when sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["harmful_k00", "harmful_k03", "benign_k00", "harmful_k10"],
    )
    return parser.parse_args()


def _load_classified(generations_dir: Path, condition: str) -> list[dict]:
    path = generations_dir / condition / "classified.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _output_text(record: dict) -> str:
    for key in ("response", "generated_text", "output", "text", "completion"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _stratified_sample(records: list[dict], condition: str, per_cell: int, rng: Random) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        label = record.get("refusal_phrase_label", "error")
        if label in ("refusal", "compliance"):
            by_label[label].append(record)
    sampled: list[dict] = []
    for label, items in by_label.items():
        if not items:
            continue
        rng.shuffle(items)
        for record in items[:per_cell]:
            sampled.append(
                {
                    "prompt_id": record.get("prompt_id"),
                    "condition": condition,
                    "auto_label": label,
                    "matched_phrase": record.get("matched_phrase"),
                    "prompt_text": record.get("prompt_text") or record.get("text") or "",
                    "output_text": _output_text(record),
                    "your_label": "TODO",
                    "notes": "",
                }
            )
    return sampled


def cmd_generate(args: argparse.Namespace) -> int:
    ensure_dir(args.out_dir)
    template_path = Path(args.template)
    if template_path.exists():
        print(f"Refusing to overwrite existing template: {template_path}")
        print("Edit it directly, or delete it and re-run `generate`.")
        return 1

    rng = Random(args.seed)
    generations_dir = Path(args.generations_dir)
    items: list[dict] = []
    for condition in args.conditions:
        records = _load_classified(generations_dir, condition)
        if not records:
            print(f"WARN: no classified records for {condition} (skipped).")
            continue
        items.extend(_stratified_sample(records, condition, args.per_cell, rng))

    rng.shuffle(items)
    for idx, item in enumerate(items, start=1):
        item["id"] = idx

    payload = {
        "instructions": (
            "Replace each `your_label: TODO` with one of: refusal, compliance, ambiguous. "
            "Then run: python scripts/run_classifier_spot_check.py score"
        ),
        "valid_labels": sorted(VALID_HUMAN_LABELS),
        "items": items,
    }
    template_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10**6))
    print(f"Wrote {len(items)} items to {template_path}")
    print("Edit the file, then re-run `python scripts/run_classifier_spot_check.py score`.")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Missing template: {template_path}. Run `generate` first.")
        return 1
    payload = yaml.safe_load(template_path.read_text())
    items = payload.get("items") or []
    if not items:
        print("No items in template.")
        return 1

    todo: list[dict] = []
    invalid: list[dict] = []
    rows: list[dict] = []
    for item in items:
        your = (item.get("your_label") or "").strip().lower()
        auto = (item.get("auto_label") or "").strip().lower()
        if your in ("", "todo"):
            todo.append(item)
            continue
        if your not in VALID_HUMAN_LABELS:
            invalid.append(item)
            continue
        # Map "ambiguous" specially: counts as a flip if auto label was decisive.
        agreement = your == auto and your != "ambiguous"
        flipped = your != auto and your != "ambiguous"
        rows.append(
            {
                "id": item.get("id"),
                "prompt_id": item.get("prompt_id"),
                "condition": item.get("condition"),
                "auto_label": auto,
                "human_label": your,
                "matched_phrase": item.get("matched_phrase"),
                "agreement": agreement,
                "flipped": flipped,
                "ambiguous": your == "ambiguous",
                "notes": item.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows)
    ensure_dir(args.out_dir)
    if df.empty:
        Path(args.csv).write_text("")
        Path(args.report).write_text("# Classifier Spot-Check\n\nNo items scored yet.\n")
        print("No scored items.")
        if todo:
            print(f"  {len(todo)} items still marked TODO.")
        return 1

    df.to_csv(args.csv, index=False)

    overall_n = len(df)
    n_flipped = int(df["flipped"].sum())
    n_ambiguous = int(df["ambiguous"].sum())
    n_agreement = int(df["agreement"].sum())
    decisive = df[~df["ambiguous"]]
    flip_rate = float(decisive["flipped"].mean()) if not decisive.empty else float("nan")

    lines = [
        "# Classifier Spot-Check Report",
        "",
        f"- Items scored: **{overall_n}**",
        f"- Auto-vs-human agreement: **{n_agreement} / {overall_n}** "
        f"({n_agreement / overall_n:.1%})",
        f"- Flips on decisive items: **{n_flipped} / {len(decisive)}** "
        f"({flip_rate:.1%})",
        f"- Ambiguous (counted neither way): **{n_ambiguous}**",
    ]
    if todo:
        lines.append(f"- TODO (not yet labeled): **{len(todo)}** — re-run `score` after filling them in.")
    if invalid:
        lines.append(f"- INVALID labels (need fixing): **{len(invalid)}**")

    lines.extend(["", "## Per-condition", ""])
    per_cond = (
        df.groupby("condition")
        .agg(
            n=("id", "size"),
            n_agreement=("agreement", "sum"),
            n_flipped=("flipped", "sum"),
            n_ambiguous=("ambiguous", "sum"),
        )
        .reset_index()
    )
    per_cond["agreement_rate"] = per_cond["n_agreement"] / per_cond["n"]
    per_cond["flip_rate"] = per_cond["n_flipped"] / per_cond["n"]
    lines.append(per_cond.to_markdown(index=False, floatfmt=".2f"))

    flips = df[df["flipped"]]
    if not flips.empty:
        lines.extend(["", "## Flipped items (human disagrees with auto)", ""])
        for _, row in flips.iterrows():
            lines.append(
                f"- id={row['id']} {row['condition']} {row['prompt_id']}: "
                f"auto=**{row['auto_label']}** human=**{row['human_label']}** "
                f"matched_phrase={row['matched_phrase']!r}"
            )
        lines.append("")

    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.report}")
    print(
        f"Summary: agreement={n_agreement}/{overall_n} "
        f"flip_rate={flip_rate:.1%} ambiguous={n_ambiguous} todo={len(todo)}"
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "generate":
        return cmd_generate(args)
    return cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
