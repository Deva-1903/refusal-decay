"""
verify_report6_outputs.py

Quick audit of Report 6 output files. Run this before starting to write the report
to confirm which experiments are complete, which have errors, and whether behavioral
data is usable.

Usage:
    python scripts/verify_report6_outputs.py

No arguments required. Reads from outputs/report6/ by default.

Exit codes:
    0 — all must-have outputs are valid
    1 — one or more must-have outputs are missing or contain only error records
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
R6 = ROOT / "outputs" / "report6"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}" + (f": {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_generation() -> bool:
    section("BEHAVIORAL GENERATION")
    conditions = ["harmful_k00", "harmful_k03", "benign_k00"]
    optional = ["harmful_k10"]
    all_ok = True

    for cond in conditions:
        cpath = R6 / "generations" / cond / "classified.jsonl"
        records = load_jsonl(cpath)
        if not records:
            ok = check(f"{cond}/classified.jsonl exists", False, "file missing or empty")
            all_ok = False
            continue

        errors = sum(1 for r in records if r.get("error"))
        valid = len(records) - errors
        refused = sum(1 for r in records if r.get("refusal_phrase_label") == "refusal")
        complied = sum(1 for r in records if r.get("refusal_phrase_label") == "compliance")

        if errors == len(records):
            ok = check(
                f"{cond}: {len(records)} records",
                False,
                f"ALL {errors} records are errors — generation failed (likely wrong GPU)",
            )
            all_ok = False
        elif errors > 0:
            ok = check(
                f"{cond}: {len(records)} records",
                False,
                f"{errors}/{len(records)} errors, {valid} valid (refusal={refused}, compliance={complied})",
            )
            all_ok = False
        else:
            refusal_rate = refused / len(records) if len(records) > 0 else 0
            ok = check(
                f"{cond}: {len(records)} records",
                True,
                f"refusal={refused}, compliance={complied}, rate={refusal_rate:.1%}",
            )

    for cond in optional:
        cpath = R6 / "generations" / cond / "classified.jsonl"
        records = load_jsonl(cpath)
        if not records:
            print(f"  [{INFO}] {cond}/classified.jsonl: missing (optional, not required for Report 6)")
            continue
        errors = sum(1 for r in records if r.get("error"))
        if errors == len(records):
            print(f"  [{WARN}] {cond}: all {len(records)} records are errors (optional condition)")
        else:
            refused = sum(1 for r in records if r.get("refusal_phrase_label") == "refusal")
            print(f"  [{INFO}] {cond}: {len(records)} records, refusal={refused} (optional condition)")

    return all_ok


def check_direction() -> bool:
    section("REFUSAL DIRECTION")
    dpath = R6 / "directions" / "refusal_direction.pt"
    if not dpath.exists():
        check("refusal_direction.pt exists", False, "file missing")
        return False

    import torch
    d = torch.load(dpath, map_location="cpu")
    if not isinstance(d, dict) or "directions" not in d:
        check("refusal_direction.pt structure", False, f"unexpected type: {type(d)}")
        return False

    dirs = d["directions"]
    expected_layers = {"16", "20", "24", "27"}
    found_layers = set(str(k) for k in dirs.keys())
    layers_ok = expected_layers.issubset(found_layers)
    check(
        f"Layers present {sorted(found_layers)}",
        layers_ok,
        "" if layers_ok else f"expected {sorted(expected_layers)}",
    )

    all_ok = layers_ok
    for layer_key, vec in dirs.items():
        norm = vec.float().norm().item()
        norm_ok = 0.95 < norm < 1.05
        check(
            f"Layer {layer_key}: shape={tuple(vec.shape)}, norm={norm:.4f}",
            norm_ok,
            "" if norm_ok else "norm far from 1.0 — direction may not be normalized",
        )
        if not norm_ok:
            all_ok = False

    metadata = d.get("metadata", {})
    print(f"  [{INFO}] Metadata: {metadata}")
    return all_ok


def check_tracing() -> bool:
    section("TRACING PARQUETS")
    try:
        import pandas as pd
    except ImportError:
        print(f"  [{WARN}] pandas not available — skipping parquet checks")
        return True

    required_k = [0, 3]
    optional_k = [10]
    required_layers = {16, 20, 24, 27}
    all_ok = True

    for k in required_k + optional_k:
        label = "k{:02d}".format(k)
        tpath = R6 / "traces" / f"traces_{label}.parquet"
        required = k in required_k

        if not tpath.exists():
            ok = check(f"traces_{label}.parquet exists", not required, "missing" + (" (optional)" if not required else ""))
            if required:
                all_ok = False
            continue

        try:
            df = pd.read_parquet(tpath)
        except Exception as e:
            ok = check(f"traces_{label}.parquet readable", False, str(e))
            if required:
                all_ok = False
            continue

        found_layers = set(df["layer"].unique())
        layers_ok = required_layers.issubset(found_layers)
        n_prompts = df["prompt_id"].nunique()
        mean_proj = df["projection"].mean()

        ok = check(
            f"traces_{label}: {len(df)} rows, {n_prompts} prompts, layers={sorted(found_layers)}",
            layers_ok and n_prompts >= 10,
            f"mean_projection={mean_proj:.4f}",
        )
        if required and not ok:
            all_ok = False

    all_path = R6 / "traces" / "traces_all.parquet"
    if all_path.exists():
        try:
            df_all = pd.read_parquet(all_path)
            check(f"traces_all.parquet: {len(df_all)} rows", True, f"conditions={sorted(df_all['prefix_k'].unique().tolist())}")
        except Exception as e:
            check("traces_all.parquet readable", False, str(e))
            all_ok = False
    else:
        check("traces_all.parquet exists", False, "missing")
        all_ok = False

    return all_ok


def check_patching() -> bool:
    section("PATCHING OUTPUTS")
    expected_combos = [
        (16, 1), (16, 3), (16, 5),
        (24, 1), (24, 3), (24, 5),
        (27, 1), (27, 3), (27, 5),
    ]
    all_ok = True

    for layer, tt in expected_combos:
        fname = f"patch_layer{layer}_ts-1_tt{tt}.jsonl"
        fpath = R6 / "patching" / fname
        records = load_jsonl(fpath)
        errors = sum(1 for r in records if r.get("error"))
        has_text = sum(1 for r in records if r.get("baseline_text") and r.get("patched_text"))
        ok = check(
            f"{fname}: n={len(records)}, has_text={has_text}",
            len(records) > 0 and errors == 0 and has_text == len(records),
            f"{errors} errors" if errors else "",
        )
        if not ok:
            all_ok = False

    classified_path = R6 / "patching" / "patching_classified.jsonl"
    records = load_jsonl(classified_path)
    if not records:
        check("patching_classified.jsonl exists and non-empty", False)
        return False

    baseline_labels = Counter(r.get("baseline_phrase_label") for r in records)
    patched_labels = Counter(r.get("patched_phrase_label") for r in records)
    restored = sum(1 for r in records if r.get("refusal_restored"))
    lost = sum(1 for r in records if r.get("refusal_lost"))

    check(
        f"patching_classified: {len(records)} records",
        len(records) == len(expected_combos) * 10,
        f"expected {len(expected_combos) * 10}",
    )
    print(f"  [{INFO}] Baseline labels: {dict(baseline_labels)}")
    print(f"  [{INFO}] Patched labels:  {dict(patched_labels)}")
    print(f"  [{INFO}] refusal_restored={restored}, refusal_lost={lost}")

    summary_path = R6 / "patching" / "patching_summary.csv"
    check("patching_summary.csv exists", summary_path.exists())

    return all_ok


def check_summaries() -> bool:
    section("SUMMARY CSVS")
    required = [
        "generation_refusal_rates_by_condition.csv",
        "trace_mean_projection_by_condition_layer.csv",
        "trace_projection_by_condition_layer_token.csv",
        "patching_refusal_recovery_summary.csv",
        "patching_prompt_results.csv",
    ]
    all_ok = True
    import csv

    for fname in required:
        fpath = R6 / "summaries" / fname
        if not fpath.exists():
            check(fname, False, "missing")
            all_ok = False
            continue

        with open(fpath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Special check: refusal rate CSV may have all NaN/empty rates
        if "refusal_rates" in fname:
            empty_rates = sum(1 for r in rows if not r.get("refusal_rate") or r.get("refusal_rate") in ("", "nan"))
            if empty_rates == len(rows):
                check(
                    fname,
                    False,
                    f"ALL {len(rows)} rows have empty refusal_rate — generation failed",
                )
                all_ok = False
            else:
                check(fname, True, f"{len(rows)} rows, {empty_rates} with empty rate")
        else:
            check(fname, len(rows) > 0, f"{len(rows)} rows")
            if not rows:
                all_ok = False

    return all_ok


def check_plots() -> bool:
    section("PLOTS")
    required = [
        ("report6_heatmap_baseline_vs_prefill.png", 10_000),
        ("report6_projection_vs_token_layer27.png", 10_000),
        ("report6_projection_vs_token_layer16.png", 10_000),
        ("report6_patching_comparison.png", 10_000),
    ]
    optional = [
        ("report6_refusal_rate_by_condition.png", 5_000),
        ("report6_prompt_level_layer27_advbench_0019.png", 5_000),
    ]

    all_ok = True
    for fname, min_bytes in required:
        fpath = R6 / "plots" / fname
        size = fpath.stat().st_size if fpath.exists() else 0
        ok = check(
            fname,
            fpath.exists() and size >= min_bytes,
            f"{size // 1024}KB" if fpath.exists() else "missing",
        )
        if not ok:
            all_ok = False

    for fname, min_bytes in optional:
        fpath = R6 / "plots" / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size < min_bytes:
                print(f"  [{WARN}] {fname}: exists but small ({size // 1024}KB) — may be blank (check if refusal_rate CSV is valid)")
            else:
                print(f"  [{INFO}] {fname}: {size // 1024}KB (optional, present)")
        else:
            print(f"  [{INFO}] {fname}: missing (optional)")

    return all_ok


def main() -> int:
    print("\nReport 6 Output Verification")
    print(f"Checking under: {R6}\n")

    results = {
        "generation": check_generation(),
        "direction": check_direction(),
        "tracing": check_tracing(),
        "patching": check_patching(),
        "summaries": check_summaries(),
        "plots": check_plots(),
    }

    section("FINAL SUMMARY")
    all_pass = all(results.values())
    for block, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  [{status}] {block.upper()}")

    if all_pass:
        print(f"\n  All must-have outputs verified. Ready to write Report 6.\n")
        return 0
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED blocks: {', '.join(failed)}")
        print("  See VERIFY_REPORT6.md for the exact commands to fix these.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
