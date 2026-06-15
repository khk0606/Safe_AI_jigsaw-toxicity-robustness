#!/usr/bin/env python3
"""Evaluate saved classical K-MHaS classifiers on the aligned benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ensure_output_dirs,
    metrics_by_variant,
)


SYSTEMS = {
    "char_tfidf_mlp": {
        "path": Path("models/kmhas_glyph_mix/char_tfidf_mlp_clean_only.joblib"),
        "text_col": "text",
    },
    "normalization_char_tfidf_mlp": {
        "path": Path("models/kmhas_glyph_mix/defense_models/normalization_char_tfidf_mlp.joblib"),
        "text_col": "normalized_text",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DATA_ROOT / "benchmark_test_2500.csv",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    frame = pd.read_csv(args.benchmark)
    metric_frames = []

    for system, config in SYSTEMS.items():
        model = joblib.load(config["path"])
        probabilities = model.predict_proba(frame[config["text_col"]].fillna("").astype(str))[:, 1]
        predictions = frame[
            ["split", "id", "variant", "label", "text", "normalized_text"]
        ].copy()
        predictions["system"] = system
        predictions["score"] = probabilities
        predictions["threshold"] = 0.5
        predictions["pred_label"] = (probabilities >= 0.5).astype(int)
        predictions["raw_output"] = predictions["pred_label"].map(
            {0: "non-hate", 1: "hate"}
        )
        predictions.to_csv(PREDICTIONS_ROOT / f"{system}.csv", index=False)
        metric_frames.append(metrics_by_variant(predictions, system))

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(REPORTS_ROOT / "classical_model_metrics.csv", index=False)
    metadata = {
        "benchmark": str(args.benchmark),
        "systems": {
            name: {"model_path": str(config["path"]), "input_column": config["text_col"]}
            for name, config in SYSTEMS.items()
        },
    }
    (REPORTS_ROOT / "classical_model_run_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics[["system", "variant", "balanced_accuracy", "macro_f1", "fnr", "fpr"]])


if __name__ == "__main__":
    main()
