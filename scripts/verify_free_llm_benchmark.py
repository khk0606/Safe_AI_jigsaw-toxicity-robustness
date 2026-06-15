#!/usr/bin/env python3
"""Run integrity checks for benchmark data and completed predictions."""

from __future__ import annotations

import json

import pandas as pd

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    VARIANT_ORDER,
)


def id_set(path) -> set[str]:
    return set(pd.read_csv(path)["id"].astype(str))


def main() -> None:
    test = pd.read_csv(DATA_ROOT / "benchmark_test_2500.csv")
    calibration = pd.read_csv(DATA_ROOT / "benchmark_calibration_2500.csv")
    expected_rows = 500 * len(VARIANT_ORDER)

    checks = {
        "test_rows": len(test) == expected_rows,
        "calibration_rows": len(calibration) == expected_rows,
        "test_unique_variant_ids": not test.duplicated(["variant", "id"]).any(),
        "calibration_unique_variant_ids": not calibration.duplicated(["variant", "id"]).any(),
        "test_label_consistency": test.groupby("id")["label"].nunique().max() == 1,
        "calibration_label_consistency": calibration.groupby("id")["label"].nunique().max() == 1,
        "test_text_nonempty": test["text"].fillna("").str.len().gt(0).all(),
        "test_normalized_nonempty": test["normalized_text"].fillna("").str.len().gt(0).all(),
    }
    for variant in VARIANT_ORDER:
        sub = test[test["variant"] == variant]
        checks[f"{variant}_rows_500"] = len(sub) == 500
        checks[f"{variant}_balanced"] = int(sub["label"].sum()) == 250

    for split in ("train", "valid"):
        lora_examples = pd.read_csv(DATA_ROOT / f"qwen_lora/{split}_examples.csv")
        checks[f"lora_{split}_max_tokens_256"] = (
            int(lora_examples["lora_token_count"].max()) <= 256
        )
        checks[f"lora_{split}_token_count_positive"] = (
            int(lora_examples["lora_token_count"].min()) > 0
        )
        checks[f"lora_{split}_supervised_tokens_positive"] = (
            int(lora_examples["lora_supervised_token_count"].min()) > 0
        )

    sets = {
        "test": id_set(DATA_ROOT / "benchmark_test_ids.csv"),
        "calibration": id_set(DATA_ROOT / "benchmark_calibration_ids.csv"),
        "lora_train": id_set(DATA_ROOT / "qwen_lora/train_examples.csv"),
        "lora_valid": id_set(DATA_ROOT / "qwen_lora/valid_examples.csv"),
    }
    names = list(sets)
    overlap = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            count = len(sets[left] & sets[right])
            overlap[f"{left}_{right}"] = count
            checks[f"no_overlap_{left}_{right}"] = count == 0

    prediction_checks = {}
    expected_pairs = set(zip(test["variant"].astype(str), test["id"].astype(str)))
    supplementary_path = DATA_ROOT / "ui_supplementary_250.csv"
    supplementary_pairs = set()
    if supplementary_path.exists():
        supplementary = pd.read_csv(supplementary_path)
        supplementary_pairs = set(
            zip(
                supplementary["variant"].astype(str),
                supplementary["id"].astype(str),
            )
        )
    for path in sorted(PREDICTIONS_ROOT.glob("*.csv")):
        frame = pd.read_csv(path)
        if not {"id", "variant", "pred_label"}.issubset(frame.columns):
            continue
        if set(frame["variant"].dropna()) == set(VARIANT_ORDER):
            actual_pairs = set(
                zip(frame["variant"].astype(str), frame["id"].astype(str))
            )
            if actual_pairs == expected_pairs:
                scope = "main_2500"
                matches_expected_scope = True
            elif supplementary_pairs and actual_pairs == supplementary_pairs:
                scope = "supplementary_250"
                matches_expected_scope = True
            else:
                scope = "unknown"
                matches_expected_scope = False
            prediction_checks[path.name] = {
                "rows": len(frame),
                "scope": scope,
                "unique_variant_ids": not frame.duplicated(["variant", "id"]).any(),
                "matches_expected_scope": matches_expected_scope,
                "parse_failures": int(frame["pred_label"].isna().sum()),
            }
            checks[f"prediction_{path.stem}_matches_expected_scope"] = (
                matches_expected_scope
                and not frame.duplicated(["variant", "id"]).any()
            )

    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "split_overlap": overlap,
        "completed_prediction_files": prediction_checks,
    }
    output = REPORTS_ROOT / "benchmark_integrity_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
