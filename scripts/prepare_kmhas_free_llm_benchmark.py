#!/usr/bin/env python3
"""Prepare aligned benchmark, calibration, LoRA, and UI subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    ROOT,
    VARIANT_ORDER,
    aligned_variant_frame,
    build_messages,
    ensure_output_dirs,
    load_variant,
)
from train_evaluate_kmhas_defense_models import normalize_obfuscated_text


QWEN_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MAX_TOKENS = 256


def balanced_ids(
    split: str,
    rows: int,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[str]:
    clean = load_variant("clean", split)
    if excluded_ids:
        clean = clean[~clean["id"].astype(str).isin(excluded_ids)].copy()
    hate_rows = rows // 2
    hate = clean[clean["label"] == 1].sample(n=hate_rows, random_state=seed)
    non_hate = clean[clean["label"] == 0].sample(n=rows - hate_rows, random_state=seed)
    sampled = (
        pd.concat([hate, non_hate], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled["id"].astype(str).tolist()


def add_normalized_text(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["normalized_text"] = out["text"].map(normalize_obfuscated_text)
    return out


def write_ui_batches(test_frame: pd.DataFrame, ui_rows_per_variant: int, batch_size: int) -> None:
    ui_root = DATA_ROOT / "ui_batches"
    ui_root.mkdir(parents=True, exist_ok=True)
    sampled_rows = []
    for variant in VARIANT_ORDER:
        sub = test_frame[test_frame["variant"] == variant].copy()
        hate = sub[sub["label"] == 1].head(ui_rows_per_variant // 2)
        non_hate = sub[sub["label"] == 0].head(ui_rows_per_variant - len(hate))
        sampled = (
            pd.concat([hate, non_hate], ignore_index=True)
            .sort_values("sample_order")
            .reset_index(drop=True)
        )
        sampled_rows.append(sampled)
        for batch_idx, start in enumerate(range(0, len(sampled), batch_size), start=1):
            batch = sampled.iloc[start : start + batch_size][["id", "variant", "text"]].copy()
            batch.to_csv(ui_root / f"{variant}_batch_{batch_idx:02d}.csv", index=False)
            jsonl = "\n".join(
                json.dumps(
                    {"id": row["id"], "variant": row["variant"], "text": row["text"]},
                    ensure_ascii=False,
                )
                for row in batch.to_dict("records")
            )
            (ui_root / f"{variant}_batch_{batch_idx:02d}.jsonl").write_text(
                jsonl + "\n", encoding="utf-8"
            )

    ui_manifest = pd.concat(sampled_rows, ignore_index=True)
    ui_manifest.to_csv(DATA_ROOT / "ui_supplementary_250.csv", index=False)

    instructions = """# ChatGPT Free / Claude Free supplementary evaluation

이 폴더의 각 JSONL batch를 새 대화에 붙여 넣고 다음 prompt를 사용한다.

```text
아래 각 한국어 댓글을 hate 또는 non-hate로 분류하세요.
부정적인 의견이더라도 개인/집단에 대한 공격·비하·적대가 없으면 non-hate입니다.
설명은 하지 말고 입력과 같은 순서의 JSONL만 출력하세요.
각 줄 형식: {"id":"...","variant":"입력의 variant","label":"hate 또는 non-hate"}
```

실행 시 다음 metadata를 기록한다.

- service: ChatGPT Free / ChatGPT Plus / Claude Free
- displayed model or mode
- run date
- fallback or rate-limit message
- batch filename

결과는 `scripts/import_ui_llm_predictions.py`로 통합한다.
"""
    (ui_root / "README.md").write_text(instructions, encoding="utf-8")


def clip_head_tail(text: str, char_budget: int) -> str:
    if len(text) <= char_budget:
        return text
    head = char_budget // 2
    tail = char_budget - head
    return text[:head] + " ... " + text[-tail:]


def make_lora_messages(
    tokenizer,
    original: str,
    normalized: str,
    label: int,
    max_tokens: int,
) -> tuple[list[dict[str, str]], int, int, bool]:
    def count_tokens(messages: list[dict[str, str]]) -> int:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        return len(encoded)

    def messages_for_budget(char_budget: int) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Korean hate-speech classifier. "
                    "Return only 1 for hate or 0 for non-hate."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify the comment. It may contain romanized Korean or rune-like glyphs. "
                    "The normalized candidate was produced by rules and may be imperfect. "
                    "Use both versions. Return only 0 or 1.\n"
                    "0 = non-hate, 1 = hate.\n\n"
                    f"Original: {clip_head_tail(original, char_budget)}\n"
                    f"Normalized candidate: {clip_head_tail(normalized, char_budget)}\n"
                    "Label:"
                ),
            },
            {"role": "assistant", "content": "1" if label == 1 else "0"},
        ]
        return messages

    maximum_chars = max(len(original), len(normalized), 16)
    low, high = 16, maximum_chars
    best_messages = messages_for_budget(low)
    best_tokens = count_tokens(best_messages)
    if best_tokens > max_tokens:
        raise ValueError("LoRA prompt template exceeds max tokens even at minimum text")

    while low <= high:
        middle = (low + high) // 2
        candidate = messages_for_budget(middle)
        token_count = count_tokens(candidate)
        if token_count <= max_tokens:
            best_messages = candidate
            best_tokens = token_count
            low = middle + 1
        else:
            high = middle - 1
    truncated = len(original) > high or len(normalized) > high
    prompt_tokens = tokenizer.apply_chat_template(
        best_messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    supervised_tokens = best_tokens - len(prompt_tokens)
    if supervised_tokens <= 0:
        raise ValueError("LoRA example has no supervised assistant tokens")
    return best_messages, best_tokens, supervised_tokens, truncated


def write_lora_data(
    train_base_rows: int,
    valid_base_rows: int,
    seed: int,
    calibration_ids: set[str],
) -> tuple[set[str], set[str]]:
    lora_root = DATA_ROOT / "qwen_lora"
    lora_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
    selected_ids: dict[str, set[str]] = {}
    for split, base_rows, split_seed in (
        ("train", train_base_rows, seed),
        ("valid", valid_base_rows, seed + 1),
    ):
        excluded_ids = calibration_ids if split == "valid" else None
        ids = balanced_ids(split, base_rows, split_seed, excluded_ids=excluded_ids)
        selected_ids[split] = set(ids)
        aligned = aligned_variant_frame(split, ids)
        aligned = aligned[aligned["variant"].isin(["clean", "romanized_35", "roman_glyph_35"])]
        aligned = add_normalized_text(aligned)

        records = []
        token_counts = []
        supervised_token_counts = []
        truncation_flags = []
        for row in aligned.to_dict("records"):
            messages, token_count, supervised_tokens, truncated = make_lora_messages(
                tokenizer,
                str(row["text"]),
                str(row["normalized_text"]),
                int(row["label"]),
                LORA_MAX_TOKENS,
            )
            records.append({"messages": messages})
            token_counts.append(token_count)
            supervised_token_counts.append(supervised_tokens)
            truncation_flags.append(truncated)
        aligned["lora_token_count"] = token_counts
        aligned["lora_supervised_token_count"] = supervised_token_counts
        aligned["lora_text_truncated"] = truncation_flags
        aligned.to_csv(lora_root / f"{split}_examples.csv", index=False)
        with (lora_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return selected_ids["train"], selected_ids["valid"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--calibration-rows", type=int, default=500)
    parser.add_argument("--ui-rows-per-variant", type=int, default=50)
    parser.add_argument("--ui-batch-size", type=int, default=25)
    parser.add_argument("--lora-train-base-rows", type=int, default=4000)
    parser.add_argument("--lora-valid-base-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    ensure_output_dirs()

    test_ids = balanced_ids("test", args.test_rows, args.seed)
    calibration_ids = balanced_ids("valid", args.calibration_rows, args.seed + 1)
    test = add_normalized_text(aligned_variant_frame("test", test_ids))
    calibration = add_normalized_text(aligned_variant_frame("valid", calibration_ids))

    test.to_csv(DATA_ROOT / "benchmark_test_2500.csv", index=False)
    calibration.to_csv(DATA_ROOT / "benchmark_calibration_2500.csv", index=False)
    pd.DataFrame({"id": test_ids}).to_csv(DATA_ROOT / "benchmark_test_ids.csv", index=False)
    pd.DataFrame({"id": calibration_ids}).to_csv(
        DATA_ROOT / "benchmark_calibration_ids.csv", index=False
    )
    write_ui_batches(test, args.ui_rows_per_variant, args.ui_batch_size)
    lora_train_ids, lora_valid_ids = write_lora_data(
        args.lora_train_base_rows,
        args.lora_valid_base_rows,
        args.seed + 10,
        calibration_ids=set(calibration_ids),
    )

    calibration_id_set = set(calibration_ids)
    test_id_set = set(test_ids)
    overlap = {
        "lora_train_lora_valid": len(lora_train_ids & lora_valid_ids),
        "lora_train_calibration": len(lora_train_ids & calibration_id_set),
        "lora_train_test": len(lora_train_ids & test_id_set),
        "lora_valid_calibration": len(lora_valid_ids & calibration_id_set),
        "lora_valid_test": len(lora_valid_ids & test_id_set),
        "calibration_test": len(calibration_id_set & test_id_set),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Split leakage found: {overlap}")

    metadata = {
        "seed": args.seed,
        "variants": VARIANT_ORDER,
        "test_base_rows": args.test_rows,
        "test_total_rows": len(test),
        "calibration_base_rows": args.calibration_rows,
        "calibration_total_rows": len(calibration),
        "lora_train_base_rows": args.lora_train_base_rows,
        "lora_train_total_rows": args.lora_train_base_rows * 3,
        "lora_valid_base_rows": args.lora_valid_base_rows,
        "lora_valid_total_rows": args.lora_valid_base_rows * 3,
        "split_overlap": overlap,
        "root": str(ROOT),
    }
    (DATA_ROOT / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
