#!/usr/bin/env python3
"""Evaluate Qwen with MLX logit scoring and validation-set calibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load
from tqdm import tqdm

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ROOT,
    ensure_output_dirs,
    metrics_by_variant,
    prediction_metrics,
)


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"


def build_messages(row: dict, input_mode: str) -> list[dict[str, str]]:
    system = (
        "You are a Korean hate-speech classifier. "
        "Return only 1 for hate or 0 for non-hate."
    )
    if input_mode == "raw":
        user = (
            "Classify the Korean comment. Return only 0 or 1.\n"
            "0 = non-hate, 1 = hate.\n\n"
            f"Comment: {row['text']}\n"
            "Label:"
        )
    else:
        user = (
            "Classify the comment. It may contain romanized Korean or rune-like glyphs. "
            "The normalized candidate was produced by rules and may be imperfect. "
            "Use both versions. Return only 0 or 1.\n"
            "0 = non-hate, 1 = hate.\n\n"
            f"Original: {row['text']}\n"
            f"Normalized candidate: {row['normalized_text']}\n"
            "Label:"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_tokens(tokenizer, row: dict, input_mode: str, max_length: int) -> list[int]:
    messages = build_messages(row, input_mode)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    return tokens[-max_length:]


def batch_scores(model, tokenizer, rows: list[dict], input_mode: str, max_length: int) -> np.ndarray:
    sequences = [prompt_tokens(tokenizer, row, input_mode, max_length) for row in rows]
    max_len = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    last_positions = []
    for idx, sequence in enumerate(sequences):
        batch[idx, : len(sequence)] = sequence
        last_positions.append(len(sequence) - 1)

    logits = model(mx.array(batch))
    if hasattr(logits, "logits"):
        logits = logits.logits
    token_zero = tokenizer.encode("0", add_special_tokens=False)
    token_one = tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero) != 1 or len(token_one) != 1:
        raise RuntimeError("Expected 0 and 1 to each map to one tokenizer token")
    scores = mx.stack(
        [
            logits[idx, position, token_one[0]] - logits[idx, position, token_zero[0]]
            for idx, position in enumerate(last_positions)
        ]
    )
    return np.asarray(scores.astype(mx.float32), dtype=np.float32)


def score_frame(
    model,
    tokenizer,
    frame: pd.DataFrame,
    input_mode: str,
    batch_size: int,
    max_length: int,
    resume_path: Path,
) -> pd.DataFrame:
    completed = {}
    if resume_path.exists():
        old = pd.read_csv(resume_path)
        completed = {
            (str(row["split"]), str(row["variant"]), str(row["id"])): row
            for row in old.to_dict("records")
        }

    output_rows = []
    pending = []
    for row in frame.to_dict("records"):
        key = (str(row["split"]), str(row["variant"]), str(row["id"]))
        if key in completed:
            output_rows.append(completed[key])
        else:
            pending.append(row)

    for start in tqdm(range(0, len(pending), batch_size), desc=f"Qwen {input_mode}"):
        batch_rows = pending[start : start + batch_size]
        scores = batch_scores(model, tokenizer, batch_rows, input_mode, max_length)
        for row, score in zip(batch_rows, scores):
            output_rows.append(
                {
                    "split": row["split"],
                    "id": row["id"],
                    "variant": row["variant"],
                    "label": int(row["label"]),
                    "text": row["text"],
                    "normalized_text": row["normalized_text"],
                    "score": float(score),
                }
            )
        if len(output_rows) % 100 < batch_size:
            pd.DataFrame(output_rows).to_csv(resume_path, index=False)
        mx.clear_cache()

    out = pd.DataFrame(output_rows)
    variant_rank = {
        "clean": 0,
        "romanized_35": 1,
        "romanized_70": 2,
        "roman_glyph_35": 3,
        "roman_glyph_70": 4,
    }
    out["_variant_rank"] = out["variant"].map(variant_rank)
    out = out.sort_values(["_variant_rank", "id"]).drop(columns="_variant_rank")
    out.to_csv(resume_path, index=False)
    return out


def choose_threshold(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    scores = frame["score"].to_numpy(dtype=float)
    if len(scores) == 0:
        raise ValueError("No calibration scores")
    quantiles = np.linspace(0.0, 1.0, 401)
    candidates = np.unique(np.quantile(scores, quantiles))
    candidates = np.concatenate(
        [[scores.min() - 1e-6], candidates, [scores.max() + 1e-6]]
    )
    rows = []
    for threshold in candidates:
        evaluated = frame.copy()
        evaluated["pred_label"] = (evaluated["score"] >= threshold).astype(int)
        metrics = prediction_metrics(evaluated)
        rows.append({"threshold": float(threshold), **metrics})
    sweep = pd.DataFrame(rows).sort_values(
        ["balanced_error", "macro_f1", "max_failure_rate"],
        ascending=[True, False, True],
    )
    return float(sweep.iloc[0]["threshold"]), sweep


def apply_threshold(frame: pd.DataFrame, threshold: float, system: str) -> pd.DataFrame:
    out = frame.copy()
    out["system"] = system
    out["threshold"] = threshold
    out["pred_label"] = (out["score"] >= threshold).astype(int)
    out["raw_output"] = out["pred_label"].map({0: "non-hate", 1: "hate"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--input-mode", choices=["raw", "normalized"], default="raw")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Score calibration data and select a threshold without touching test data.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    suffix = args.run_name or (
        f"qwen_{'lora_' if args.adapter_path else ''}{args.input_mode}"
    )
    calibration_scores_path = PREDICTIONS_ROOT / f"{suffix}_calibration_scores.csv"
    test_scores_path = PREDICTIONS_ROOT / f"{suffix}_test_scores.csv"
    if not args.resume:
        for path in (calibration_scores_path, test_scores_path):
            if path.exists():
                path.unlink()

    start = time.time()
    model, tokenizer = load(
        str(args.model),
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    calibration = pd.read_csv(DATA_ROOT / "benchmark_calibration_2500.csv")
    test = pd.read_csv(DATA_ROOT / "benchmark_test_2500.csv")
    calibration_scores = score_frame(
        model,
        tokenizer,
        calibration,
        args.input_mode,
        args.batch_size,
        args.max_length,
        calibration_scores_path,
    )
    threshold, sweep = choose_threshold(calibration_scores)
    sweep.to_csv(REPORTS_ROOT / f"{suffix}_threshold_sweep.csv", index=False)
    calibration_best = sweep.iloc[[0]].copy()
    calibration_best.insert(0, "run_name", suffix)
    calibration_best.to_csv(
        REPORTS_ROOT / f"{suffix}_calibration_selection.csv", index=False
    )
    if args.calibration_only:
        metadata = {
            "model": str(args.model),
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "input_mode": args.input_mode,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "calibrated_threshold": threshold,
            "calibration_only": True,
            "elapsed_seconds": time.time() - start,
        }
        (REPORTS_ROOT / f"{suffix}_run_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(calibration_best.to_string(index=False))
        return

    test_scores = score_frame(
        model,
        tokenizer,
        test,
        args.input_mode,
        args.batch_size,
        args.max_length,
        test_scores_path,
    )

    uncalibrated = apply_threshold(
        test_scores, 0.0, f"{suffix}_uncalibrated"
    )
    calibrated = apply_threshold(test_scores, threshold, f"{suffix}_calibrated")
    uncalibrated.to_csv(
        PREDICTIONS_ROOT / f"{suffix}_uncalibrated.csv", index=False
    )
    calibrated.to_csv(PREDICTIONS_ROOT / f"{suffix}_calibrated.csv", index=False)
    metrics = pd.concat(
        [
            metrics_by_variant(uncalibrated, f"{suffix}_uncalibrated"),
            metrics_by_variant(calibrated, f"{suffix}_calibrated"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(REPORTS_ROOT / f"{suffix}_metrics.csv", index=False)
    metadata = {
        "model": str(args.model),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "input_mode": args.input_mode,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "calibrated_threshold": threshold,
        "calibration_only": False,
        "elapsed_seconds": time.time() - start,
    }
    (REPORTS_ROOT / f"{suffix}_run_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metrics[["system", "variant", "balanced_accuracy", "macro_f1", "fnr", "fpr"]])


if __name__ == "__main__":
    main()
