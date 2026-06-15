#!/usr/bin/env python3
"""Create readable romanized/glyph-mixed K-MHaS perturbation variants.

This script uses the already standardized clean K-MHaS CSV files and creates
intermediate obfuscation levels that are easier for Korean readers to infer
than the older glyph_mix_35/glyph_mix_90 stress-test files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path


HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
JUNGSEONG_COUNT = 21
JONGSEONG_COUNT = 28

CHOSEONG = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]

JUNGSEONG = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]

JONGSEONG = [
    "",
    "g",
    "kk",
    "gs",
    "n",
    "nj",
    "nh",
    "d",
    "l",
    "lg",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "b",
    "bs",
    "s",
    "ss",
    "ng",
    "j",
    "ch",
    "k",
    "t",
    "p",
    "h",
]

LATIN_TO_GLYPH = {
    "a": "ᚨ",
    "b": "ᛒ",
    "c": "ᚲ",
    "d": "ᛞ",
    "e": "ᛖ",
    "f": "ᚠ",
    "g": "ᚷ",
    "h": "ᚺ",
    "i": "ᛁ",
    "j": "ᛃ",
    "k": "ᚴ",
    "l": "ᛚ",
    "m": "ᛗ",
    "n": "ᚾ",
    "o": "ᛟ",
    "p": "ᛈ",
    "q": "ᛩ",
    "r": "ᚱ",
    "s": "ᛋ",
    "t": "ᛏ",
    "u": "ᚢ",
    "v": "ᚡ",
    "w": "ᚹ",
    "x": "ᛪ",
    "y": "ᚣ",
    "z": "ᛉ",
}


@dataclass(frozen=True)
class Variant:
    name: str
    syllable_ratio: float
    glyph_ratio: float
    description: str


VARIANTS = [
    Variant(
        name="romanized_35",
        syllable_ratio=0.35,
        glyph_ratio=0.0,
        description="Mild readable perturbation: replace about 35% of Hangul syllables with romanized Korean.",
    ),
    Variant(
        name="romanized_70",
        syllable_ratio=0.70,
        glyph_ratio=0.0,
        description="Stronger readable perturbation: replace about 70% of Hangul syllables with romanized Korean.",
    ),
    Variant(
        name="roman_glyph_35",
        syllable_ratio=0.35,
        glyph_ratio=0.35,
        description="Mild mixed perturbation: romanize about 35% of Hangul syllables and glyph-mix some roman letters.",
    ),
    Variant(
        name="roman_glyph_70",
        syllable_ratio=0.70,
        glyph_ratio=0.45,
        description="Strong mixed perturbation: romanize about 70% of Hangul syllables and glyph-mix many roman letters.",
    ),
]


def romanize_hangul_syllable(char: str) -> str | None:
    code = ord(char)
    if not (HANGUL_BASE <= code <= HANGUL_END):
        return None

    syllable_index = code - HANGUL_BASE
    choseong_index = syllable_index // (JUNGSEONG_COUNT * JONGSEONG_COUNT)
    jungseong_index = (syllable_index % (JUNGSEONG_COUNT * JONGSEONG_COUNT)) // JONGSEONG_COUNT
    jongseong_index = syllable_index % JONGSEONG_COUNT
    return (
        CHOSEONG[choseong_index]
        + JUNGSEONG[jungseong_index]
        + JONGSEONG[jongseong_index]
    )


def glyphify_latin(text: str, rng: random.Random, glyph_ratio: float) -> str:
    out = []
    for char in text:
        lower = char.lower()
        if lower in LATIN_TO_GLYPH and rng.random() < glyph_ratio:
            out.append(LATIN_TO_GLYPH[lower])
        else:
            out.append(char)
    return "".join(out)


def obfuscate_text(text: str, rng: random.Random, variant: Variant) -> tuple[str, int, int]:
    out: list[str] = []
    hangul_syllables = 0
    replaced_syllables = 0

    for char in text:
        romanized = romanize_hangul_syllable(char)
        if romanized is None:
            out.append(char)
            continue

        hangul_syllables += 1
        if rng.random() < variant.syllable_ratio:
            replaced_syllables += 1
            out.append(glyphify_latin(romanized, rng, variant.glyph_ratio))
        else:
            out.append(char)

    return "".join(out), hangul_syllables, replaced_syllables


def visible_chars(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def glyph_char_ratio(text: str) -> float:
    chars = visible_chars(text)
    if not chars:
        return 0.0
    glyph_count = sum(char in LATIN_TO_GLYPH.values() for char in chars)
    return glyph_count / len(chars)


def latin_char_ratio(text: str) -> float:
    chars = visible_chars(text)
    if not chars:
        return 0.0
    latin_count = sum(char.isascii() and char.isalpha() for char in chars)
    return latin_count / len(chars)


def process_split(input_path: Path, output_path: Path, split: str, seed: int, variant: Variant) -> dict[str, object]:
    total = 0
    hate = 0
    glyph_ratio_sum = 0.0
    latin_ratio_sum = 0.0
    replaced_sum = 0
    hangul_sum = 0

    with input_path.open("r", encoding="utf-8", newline="") as f_in, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = [
            "split",
            "id",
            "document",
            "obfuscated_document",
            "label",
            "is_hate_speech",
            "perturbation_type",
            "syllable_obfuscation_ratio_target",
            "glyph_letter_ratio_target",
            "observed_syllable_replacement_ratio",
            "observed_latin_char_ratio",
            "observed_glyph_char_ratio",
            "source",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            total += 1
            hate += int(row["is_hate_speech"])
            text = row["document"]
            row_seed_text = f"{seed}:{variant.name}:{split}:{idx}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            row_rng = random.Random(row_seed_text)
            obfuscated, hangul_count, replaced_count = obfuscate_text(text, row_rng, variant)

            hangul_sum += hangul_count
            replaced_sum += replaced_count
            glyph_ratio = glyph_char_ratio(obfuscated)
            latin_ratio = latin_char_ratio(obfuscated)
            glyph_ratio_sum += glyph_ratio
            latin_ratio_sum += latin_ratio
            syllable_replacement_ratio = replaced_count / hangul_count if hangul_count else 0.0

            writer.writerow(
                {
                    "split": split,
                    "id": row["id"],
                    "document": text,
                    "obfuscated_document": obfuscated,
                    "label": row["label"],
                    "is_hate_speech": row["is_hate_speech"],
                    "perturbation_type": variant.name,
                    "syllable_obfuscation_ratio_target": variant.syllable_ratio,
                    "glyph_letter_ratio_target": variant.glyph_ratio,
                    "observed_syllable_replacement_ratio": f"{syllable_replacement_ratio:.6f}",
                    "observed_latin_char_ratio": f"{latin_ratio:.6f}",
                    "observed_glyph_char_ratio": f"{glyph_ratio:.6f}",
                    "source": row.get("source", "adlnlp/K-MHaS"),
                }
            )

    return {
        "variant": variant.name,
        "split": split,
        "rows": total,
        "hate_rows": hate,
        "non_hate_rows": total - hate,
        "target_syllable_ratio": variant.syllable_ratio,
        "target_glyph_ratio": variant.glyph_ratio,
        "mean_observed_syllable_replacement_ratio": round(replaced_sum / hangul_sum, 6) if hangul_sum else 0.0,
        "mean_observed_latin_char_ratio": round(latin_ratio_sum / total, 6) if total else 0.0,
        "mean_observed_glyph_char_ratio": round(glyph_ratio_sum / total, 6) if total else 0.0,
    }


def combine_variant_files(split_paths: list[Path], combined_path: Path) -> None:
    with combined_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = None
        for path in split_paths:
            with path.open("r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                if writer is None:
                    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)


def write_readme(output_root: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "# K-MHaS Readable Romanized Obfuscation Variants",
        "",
        "이 데이터셋은 K-MHaS clean split에서 출발해, 사람이 어느 정도 추론 가능한 난독화 단계를 추가로 만든 버전입니다.",
        "",
        "## Transformation Direction",
        "",
        "```text",
        "한국어 원문",
        "-> romanized Korean replacement",
        "-> romanized Korean + rune/glyph letter mixing",
        "```",
        "",
        "예시:",
        "",
        "```text",
        "부산의 자랑",
        "-> busanui jarang",
        "-> ᛒᚢsanui jᚨrang",
        "```",
        "",
        "## Variants",
        "",
        "| Variant | Meaning | Intended role |",
        "|---|---|---|",
        "| romanized_35 | 약 35% 한글 음절을 로마자 발음 표기로 대치 | mild/readable obfuscation |",
        "| romanized_70 | 약 70% 한글 음절을 로마자 발음 표기로 대치 | stronger readable obfuscation |",
        "| roman_glyph_35 | 약 35% 음절 로마자화 + 로마자 일부 glyph 치환 | mild mixed obfuscation |",
        "| roman_glyph_70 | 약 70% 음절 로마자화 + 더 많은 glyph 치환 | strong mixed obfuscation |",
        "",
        "## Label Rule",
        "",
        "원본 K-MHaS label을 유지하며 binary task는 다음 규칙을 사용합니다.",
        "",
        "```text",
        "label == 8  -> non-hate",
        "label != 8  -> hate",
        "```",
        "",
        "## Summary",
        "",
        "| Variant | Split | Rows | Hate | Non-hate | Mean replaced syllables | Mean latin chars | Mean glyph chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {variant} | {split} | {rows} | {hate_rows} | {non_hate_rows} | {mean_observed_syllable_replacement_ratio:.4f} | {mean_observed_latin_char_ratio:.4f} | {mean_observed_glyph_char_ratio:.4f} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "각 variant 디렉토리에는 train/valid/test CSV와 all_splits CSV가 포함됩니다.",
            "",
            "```text",
            "kmhas_train_<variant>.csv",
            "kmhas_valid_<variant>.csv",
            "kmhas_test_<variant>.csv",
            "kmhas_all_splits_<variant>.csv",
            "```",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def zip_output(output_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_root.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, default=Path("data/kmhas/processed/clean"))
    parser.add_argument("--output-root", type=Path, default=Path("data/kmhas/processed/readable_obfuscation"))
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/kmhas/processed/kmhas_readable_obfuscation_variants.zip"),
    )
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_dir = args.output_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        split_paths = []

        for split in ["train", "valid", "test"]:
            input_path = args.clean_root / f"kmhas_{split}_clean.csv"
            output_path = variant_dir / f"kmhas_{split}_{variant.name}.csv"
            split_paths.append(output_path)
            summary.append(process_split(input_path, output_path, split, args.seed, variant))

        combine_variant_files(split_paths, variant_dir / f"kmhas_all_splits_{variant.name}.csv")

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(args.output_root, summary)
    zip_output(args.output_root, args.zip_path)

    print(f"Saved readable obfuscation datasets to {args.output_root}")
    print(f"Saved zip to {args.zip_path}")


if __name__ == "__main__":
    main()
