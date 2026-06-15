#!/usr/bin/env python3
"""Evaluate Gemini free-tier API on the aligned K-MHaS benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ensure_output_dirs,
    metrics_by_variant,
    parse_label,
)


def request_batch(
    api_key: str,
    model: str,
    rows: list[dict],
    timeout: int,
) -> list[dict]:
    items = [
        {"id": str(row["id"]), "variant": str(row["variant"]), "text": str(row["text"])}
        for row in rows
    ]
    prompt = (
        "Classify each Korean comment as hate or non-hate. "
        "Hate includes attacks, insults, dehumanization, or hostility toward a person "
        "or social group. A negative opinion without such attacks is non-hate. "
        "Return one result for every input id.\n\n"
        f"Inputs:\n{json.dumps(items, ensure_ascii=False)}"
    )
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "STRING"},
                "label": {"type": "STRING", "enum": ["hate", "non-hate"]},
            },
            "required": ["id", "label"],
        },
    }
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a strict Korean hate-speech classifier. "
                        "Do not refuse. Classify only the provided comments."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max(256, len(rows) * 20),
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent?key="
        f"{urllib.parse.quote(api_key, safe='')}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing {args.api_key_env}. Create a free Gemini API key in Google AI Studio "
            f"and export {args.api_key_env}=..."
        )

    benchmark = pd.read_csv(DATA_ROOT / "benchmark_test_2500.csv")
    output_path = PREDICTIONS_ROOT / "gemini_2_5_flash_lite.csv"
    completed = {}
    if args.resume and output_path.exists():
        old = pd.read_csv(output_path)
        completed = {
            (str(row["variant"]), str(row["id"])): row for row in old.to_dict("records")
        }

    output_rows = list(completed.values())
    pending = [
        row
        for row in benchmark.to_dict("records")
        if (str(row["variant"]), str(row["id"])) not in completed
    ]
    for start in tqdm(range(0, len(pending), args.batch_size), desc="Gemini"):
        batch = pending[start : start + args.batch_size]
        last_error = None
        results = None
        for attempt in range(args.max_retries):
            try:
                results = request_batch(api_key, args.model, batch, args.timeout)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                delay = min(60.0, (2**attempt) + random.random())
                print(f"Gemini batch failed ({exc}); retrying in {delay:.1f}s")
                time.sleep(delay)
        if results is None:
            raise RuntimeError(f"Gemini failed after retries: {last_error}")

        result_map = {str(item["id"]): item.get("label") for item in results}
        for row in batch:
            raw_output = result_map.get(str(row["id"]), "")
            pred_label, parsed_label = parse_label(raw_output)
            output_rows.append(
                {
                    "split": row["split"],
                    "id": row["id"],
                    "variant": row["variant"],
                    "label": int(row["label"]),
                    "text": row["text"],
                    "normalized_text": row["normalized_text"],
                    "system": "gemini_2_5_flash_lite",
                    "score": "",
                    "threshold": "",
                    "pred_label": pred_label,
                    "parsed_label": parsed_label,
                    "raw_output": raw_output,
                }
            )
        pd.DataFrame(output_rows).to_csv(output_path, index=False)
        time.sleep(args.sleep_seconds)

    predictions = pd.DataFrame(output_rows)
    predictions.to_csv(output_path, index=False)
    metrics = metrics_by_variant(predictions, "gemini_2_5_flash_lite")
    metrics.to_csv(REPORTS_ROOT / "gemini_2_5_flash_lite_metrics.csv", index=False)
    run_info = {
        "model": args.model,
        "rows": len(predictions),
        "batch_size": args.batch_size,
        "temperature": 0,
        "api_tier": "free",
    }
    (REPORTS_ROOT / "gemini_2_5_flash_lite_run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics[["variant", "balanced_accuracy", "macro_f1", "fnr", "fpr"]])


if __name__ == "__main__":
    main()
