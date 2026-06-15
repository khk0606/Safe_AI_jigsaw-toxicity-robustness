#!/usr/bin/env python3
"""Create a deterministic glyph-mix perturbation dataset from K-MHaS.

The output is intended for robustness evaluation. It keeps the original K-MHaS
labels and adds an obfuscated text column generated from the document column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
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

ASCII_CONFUSABLE = {
    "가": "ㄱr",
    "나": "ㄴr",
    "다": "ㄷr",
    "라": "ㄹr",
    "마": "ㅁr",
    "바": "ㅂr",
    "사": "ㅅr",
    "아": "ㅇr",
    "자": "zㅏ",
    "차": "ㅊr",
    "카": "ㅋr",
    "타": "ㅌr",
    "파": "ㅍr",
    "하": "hㅏ",
    "이": "0l",
    "오": "0",
}


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


def obfuscate_text(text: str, rng: random.Random, syllable_ratio: float, glyph_ratio: float) -> str:
    out = []
    for char in text:
        if char.isspace():
            out.append(char)
            continue

        if char in ASCII_CONFUSABLE and rng.random() < 0.15:
            out.append(ASCII_CONFUSABLE[char])
            continue

        romanized = romanize_hangul_syllable(char)
        if romanized is not None:
            if rng.random() < syllable_ratio:
                out.append(glyphify_latin(romanized, rng, glyph_ratio))
            else:
                out.append(char)
            continue

        if char.isascii() and char.isalpha() and rng.random() < syllable_ratio:
            out.append(glyphify_latin(char, rng, glyph_ratio))
        else:
            out.append(char)

    return "".join(out)


def glyph_char_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    glyph_count = sum(char in LATIN_TO_GLYPH.values() for char in visible)
    return glyph_count / len(visible)


def is_hate_label(label: str) -> int:
    labels = {item.strip() for item in label.split(",")}
    return int("8" not in labels)


def process_split(
    input_path: Path,
    output_path: Path,
    split: str,
    seed: int,
    syllable_ratio: float,
    glyph_ratio: float,
    variant_name: str,
) -> dict[str, int | float | str]:
    total = 0
    hate = 0
    ratio_sum = 0.0

    with input_path.open("r", encoding="utf-8", newline="") as f_in, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
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
            "observed_glyph_char_ratio",
            "source",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            total += 1
            label = row["label"]
            hate_flag = is_hate_label(label)
            hate += hate_flag
            text = row["document"]
            row_seed_text = f"{seed}:{split}:{idx}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            row_rng = random.Random(row_seed_text)
            obfuscated = obfuscate_text(text, row_rng, syllable_ratio, glyph_ratio)
            observed_ratio = glyph_char_ratio(obfuscated)
            ratio_sum += observed_ratio

            writer.writerow(
                {
                    "split": split,
                    "id": f"{split}_{idx:06d}",
                    "document": text,
                    "obfuscated_document": obfuscated,
                    "label": label,
                    "is_hate_speech": hate_flag,
                    "perturbation_type": variant_name,
                    "syllable_obfuscation_ratio_target": syllable_ratio,
                    "glyph_letter_ratio_target": glyph_ratio,
                    "observed_glyph_char_ratio": f"{observed_ratio:.6f}",
                    "source": "adlnlp/K-MHaS",
                }
            )

    return {
        "split": split,
        "rows": total,
        "hate_rows": hate,
        "not_hate_rows": total - hate,
        "mean_observed_glyph_char_ratio": round(ratio_sum / total, 6) if total else 0.0,
        "output": str(output_path),
    }


def write_readme(output_dir: Path, summary: list[dict[str, int | float | str]], args: argparse.Namespace) -> None:
    readme = f"""# K-MHaS {args.variant_name} Robustness Dataset

This folder contains a derived robustness-evaluation dataset built from
`adlnlp/K-MHaS`.

## Source Dataset

- Repository: https://github.com/adlnlp/K-MHaS
- Paper: K-MHaS: A Multi-label Hate Speech Detection Dataset in Korean Online News Comment
- Task labels: multi-label Korean hate speech labels.
- Label `8` means `Not Hate Speech`; any label set not containing `8` is treated as hate speech for the added binary helper column.

## Generated Columns

- `split`: train, valid, or test.
- `id`: deterministic split-local row id.
- `document`: original K-MHaS comment text.
- `obfuscated_document`: glyph-mix perturbation of `document`.
- `label`: original K-MHaS multi-label string.
- `is_hate_speech`: binary helper column, `1` if label is not `8`.
- `perturbation_type`: `{args.variant_name}`.
- `syllable_obfuscation_ratio_target`: probability of transforming each Hangul syllable.
- `glyph_letter_ratio_target`: probability of converting each romanized letter to a rune-style glyph.
- `observed_glyph_char_ratio`: realized glyph-character ratio after transformation.
- `source`: source dataset id.

## Perturbation Recipe

Each Hangul syllable is deterministically romanized from its initial consonant,
vowel, and final consonant. With target probability `{args.syllable_ratio}`, the
syllable is replaced by a romanized form whose letters are converted to
rune-style glyphs with target probability `{args.glyph_ratio}`.

The generator uses a fixed seed (`{args.seed}`) plus each row's split, index, and
text hash, so the output is reproducible while still varying row by row.

## Intended Use

This is intended for controlled robustness evaluation of Korean hate-speech
classifiers and small Korean LMs under extreme text obfuscation. Because the
source dataset contains hateful and offensive comments, avoid publishing raw
rows directly in slides or public demos. Prefer aggregate metrics and masked
examples.

## Summary

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def zip_outputs(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))


def combine_split_csvs(output_dir: Path, variant_name: str) -> Path:
    combined_path = output_dir / f"kmhas_all_splits_{variant_name}.csv"
    split_paths = [
        output_dir / f"kmhas_train_{variant_name}.csv",
        output_dir / f"kmhas_valid_{variant_name}.csv",
        output_dir / f"kmhas_test_{variant_name}.csv",
    ]

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
    return combined_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/kmhas/raw"))
    parser.add_argument("--variant-name", default="glyph_mix_90")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--syllable-ratio", type=float, default=1.00)
    parser.add_argument("--glyph-ratio", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("data/kmhas/processed") / args.variant_name
    if args.zip_path is None:
        args.zip_path = Path("data/kmhas/processed") / f"kmhas_{args.variant_name}.zip"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": args.raw_dir / "kmhas_train.txt",
        "valid": args.raw_dir / "kmhas_valid.txt",
        "test": args.raw_dir / "kmhas_test.txt",
    }

    summary = []
    for split, input_path in split_files.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing source split: {input_path}")
        output_path = args.output_dir / f"kmhas_{split}_{args.variant_name}.csv"
        summary.append(
            process_split(
                input_path,
                output_path,
                split,
                args.seed,
                args.syllable_ratio,
                args.glyph_ratio,
                args.variant_name,
            )
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "adlnlp/K-MHaS",
                "perturbation_type": args.variant_name,
                "seed": args.seed,
                "syllable_obfuscation_ratio_target": args.syllable_ratio,
                "glyph_letter_ratio_target": args.glyph_ratio,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    combined_path = combine_split_csvs(args.output_dir, args.variant_name)
    write_readme(args.output_dir, summary, args)
    zip_outputs(args.output_dir, args.zip_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"combined={combined_path}")
    print(f"zip={args.zip_path}")


if __name__ == "__main__":
    main()
