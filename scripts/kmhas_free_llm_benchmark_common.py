#!/usr/bin/env python3
"""Shared utilities for the free K-MHaS LLM robustness benchmark."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "outputs/kmhas_free_llm_benchmark"
DATA_ROOT = BENCHMARK_ROOT / "data"
PREDICTIONS_ROOT = BENCHMARK_ROOT / "predictions"
REPORTS_ROOT = BENCHMARK_ROOT / "reports"
FIGURES_ROOT = BENCHMARK_ROOT / "figures"

VARIANT_ORDER = [
    "clean",
    "romanized_35",
    "romanized_70",
    "roman_glyph_35",
    "roman_glyph_70",
]

VARIANT_LABELS = {
    "clean": "Clean",
    "romanized_35": "Romanized 35",
    "romanized_70": "Romanized 70",
    "roman_glyph_35": "Roman+Glyph 35",
    "roman_glyph_70": "Roman+Glyph 70",
}

VARIANT_PATHS = {
    "clean": {
        "train": ROOT / "data/kmhas/processed/clean/kmhas_train_clean.csv",
        "valid": ROOT / "data/kmhas/processed/clean/kmhas_valid_clean.csv",
        "test": ROOT / "data/kmhas/processed/clean/kmhas_test_clean.csv",
    },
    "romanized_35": {
        split: ROOT
        / f"data/kmhas/processed/readable_obfuscation/romanized_35/kmhas_{split}_romanized_35.csv"
        for split in ("train", "valid", "test")
    },
    "romanized_70": {
        split: ROOT
        / f"data/kmhas/processed/readable_obfuscation/romanized_70/kmhas_{split}_romanized_70.csv"
        for split in ("train", "valid", "test")
    },
    "roman_glyph_35": {
        split: ROOT
        / f"data/kmhas/processed/readable_obfuscation/roman_glyph_35/kmhas_{split}_roman_glyph_35.csv"
        for split in ("train", "valid", "test")
    },
    "roman_glyph_70": {
        split: ROOT
        / f"data/kmhas/processed/readable_obfuscation/roman_glyph_70/kmhas_{split}_roman_glyph_70.csv"
        for split in ("train", "valid", "test")
    },
}

SYSTEM_PROMPT = (
    "You are a Korean hate-speech classifier. "
    "Classify each comment according to whether it attacks, insults, dehumanizes, "
    "or expresses hostility toward a person or social group. "
    "Negative opinions without such attacks are non-hate. "
    "Return only hate or non-hate."
)


def ensure_output_dirs() -> None:
    for path in (DATA_ROOT, PREDICTIONS_ROOT, REPORTS_ROOT, FIGURES_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def text_column_for_variant(variant: str) -> str:
    return "document" if variant == "clean" else "obfuscated_document"


def load_variant(variant: str, split: str) -> pd.DataFrame:
    path = VARIANT_PATHS[variant][split]
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    text_col = text_column_for_variant(variant)
    return pd.DataFrame(
        {
            "split": split,
            "id": frame["id"].astype(str),
            "variant": variant,
            "text": frame[text_col].fillna("").astype(str),
            "original_document": frame["document"].fillna("").astype(str),
            "label": frame["is_hate_speech"].astype(int),
            "kmhas_label": frame["label"].astype(str),
        }
    )


def aligned_variant_frame(split: str, ids: list[str]) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        frame = load_variant(variant, split).set_index("id")
        missing = sorted(set(ids) - set(frame.index))
        if missing:
            raise ValueError(f"{variant}/{split} is missing {len(missing)} benchmark ids")
        selected = frame.loc[ids].reset_index()
        selected["sample_order"] = np.arange(len(selected))
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def build_messages(comment: str, normalized_comment: str | None = None) -> list[dict[str, str]]:
    if normalized_comment is None:
        user = (
            "Classify this Korean comment as hate or non-hate.\n"
            "Return exactly one label and no explanation.\n\n"
            f"Comment: {comment}\n"
            "Label:"
        )
    else:
        user = (
            "Classify this Korean comment as hate or non-hate. "
            "The original may contain romanized Korean or rune-like glyphs. "
            "A rule-based normalizer produced a possible reconstruction; it may be imperfect. "
            "Use both versions and return exactly one label with no explanation.\n\n"
            f"Original: {comment}\n"
            f"Normalized candidate: {normalized_comment}\n"
            "Label:"
        )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_label(value: Any) -> tuple[float, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, "parse_failed"
    text = str(value).strip().lower().replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    if re.search(r"\bnon\s*-\s*hate\b|\bnonhate\b|\bnon hate\b", text):
        return 0.0, "non-hate"
    if re.search(r"\bhate\b", text):
        return 1.0, "hate"
    if "비혐오" in text or "비악성" in text or text in {"정상", "일반"}:
        return 0.0, "non-hate"
    if "혐오" in text or "악성" in text:
        return 1.0, "hate"
    return np.nan, "parse_failed"


def prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[frame["pred_label"].notna()].copy()
    parse_failures = int(frame["pred_label"].isna().sum())
    if valid.empty:
        return {
            "rows": len(frame),
            "valid_predictions": 0,
            "parse_failures": parse_failures,
            "parse_failure_rate": 1.0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "hate_precision": np.nan,
            "hate_recall": np.nan,
            "fnr": np.nan,
            "fpr": np.nan,
            "predicted_hate_rate": np.nan,
            "balanced_error": np.nan,
            "max_failure_rate": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_true = valid["label"].astype(int).to_numpy()
    y_pred = valid["pred_label"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / max(1, fn + tp)
    fpr = fp / max(1, fp + tn)
    return {
        "rows": len(frame),
        "valid_predictions": len(valid),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / max(1, len(frame)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hate_precision": precision_score(y_true, y_pred, zero_division=0),
        "hate_recall": recall_score(y_true, y_pred, zero_division=0),
        "fnr": fnr,
        "fpr": fpr,
        "predicted_hate_rate": float(np.mean(y_pred)),
        "balanced_error": (fnr + fpr) / 2,
        "max_failure_rate": max(fnr, fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_by_variant(predictions: pd.DataFrame, system_name: str, split: str = "test") -> pd.DataFrame:
    frame = predictions[predictions["split"] == split].copy()
    rows = []
    clean_balanced_accuracy = None
    for variant in VARIANT_ORDER:
        sub = frame[frame["variant"] == variant]
        metrics = prediction_metrics(sub)
        if variant == "clean":
            clean_balanced_accuracy = metrics["balanced_accuracy"]
        metrics.update(
            {
                "system": system_name,
                "split": split,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
            }
        )
        if clean_balanced_accuracy is None or np.isnan(clean_balanced_accuracy):
            metrics["balanced_accuracy_drop_from_clean"] = np.nan
        else:
            metrics["balanced_accuracy_drop_from_clean"] = (
                clean_balanced_accuracy - metrics["balanced_accuracy"]
            )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_systems(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, group in metrics.groupby("system", sort=False):
        rows.append(
            {
                "system": system,
                "worst_case_balanced_accuracy": group["balanced_accuracy"].min(),
                "mean_balanced_accuracy": group["balanced_accuracy"].mean(),
                "mean_macro_f1": group["macro_f1"].mean(),
                "worst_max_failure_rate": group["max_failure_rate"].max(),
                "mean_fnr": group["fnr"].mean(),
                "mean_fpr": group["fpr"].mean(),
                "mean_predicted_hate_rate": group["predicted_hate_rate"].mean(),
                "total_parse_failures": int(group["parse_failures"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["worst_case_balanced_accuracy", "worst_max_failure_rate", "mean_balanced_accuracy"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
