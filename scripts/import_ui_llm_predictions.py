#!/usr/bin/env python3
"""Import ChatGPT/Claude web UI outputs into the common benchmark schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ensure_output_dirs,
    metrics_by_variant,
    parse_label,
)


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--displayed-model", default="unknown")
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    ensure_output_dirs()

    manifest = pd.read_csv(DATA_ROOT / "ui_supplementary_250.csv")
    results = read_results(args.input)
    if not {"id", "variant", "label"}.issubset(results.columns):
        raise ValueError("UI result file must contain id, variant, and label columns")
    results["id"] = results["id"].astype(str)
    if results.duplicated(["id", "variant"]).any():
        raise ValueError("Duplicate id/variant pairs in UI result file")

    merged = manifest.merge(
        results[["id", "variant", "label"]],
        on=["id", "variant"],
        how="left",
        suffixes=("", "_ui"),
    )
    parsed = merged["label_ui"].map(parse_label)
    merged["pred_label"] = [item[0] for item in parsed]
    merged["parsed_label"] = [item[1] for item in parsed]
    merged["raw_output"] = merged["label_ui"]
    merged["system"] = args.system
    merged["score"] = ""
    merged["threshold"] = ""
    output_path = PREDICTIONS_ROOT / f"{args.system}.csv"
    merged.to_csv(output_path, index=False)
    metrics = metrics_by_variant(merged, args.system)
    metrics.to_csv(REPORTS_ROOT / f"{args.system}_metrics.csv", index=False)
    metadata = {
        "system": args.system,
        "displayed_model": args.displayed_model,
        "run_date": args.run_date,
        "notes": args.notes,
        "input": str(args.input),
        "expected_rows": len(manifest),
        "received_rows": int(merged["pred_label"].notna().sum()),
    }
    (REPORTS_ROOT / f"{args.system}_run_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics[["variant", "balanced_accuracy", "macro_f1", "fnr", "fpr"]])


if __name__ == "__main__":
    main()
