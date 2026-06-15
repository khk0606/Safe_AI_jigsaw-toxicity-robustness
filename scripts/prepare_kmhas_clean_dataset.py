#!/usr/bin/env python3
"""Convert raw K-MHaS TSV splits into standardized clean CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def is_hate_label(label: str) -> int:
    labels = {item.strip() for item in label.split(",")}
    return int("8" not in labels)


def process_split(input_path: Path, output_path: Path, split: str) -> dict[str, int | str]:
    rows = 0
    hate_rows = 0
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
            label = row["label"]
            hate_flag = is_hate_label(label)
            writer.writerow(
                {
                    "split": split,
                    "id": f"{split}_{idx:06d}",
                    "document": row["document"],
                    "obfuscated_document": row["document"],
                    "label": label,
                    "is_hate_speech": hate_flag,
                    "perturbation_type": "clean",
                    "syllable_obfuscation_ratio_target": 0.0,
                    "glyph_letter_ratio_target": 0.0,
                    "observed_glyph_char_ratio": "0.000000",
                    "source": "adlnlp/K-MHaS",
                }
            )
            rows += 1
            hate_rows += hate_flag
    return {
        "split": split,
        "rows": rows,
        "hate_rows": hate_rows,
        "not_hate_rows": rows - hate_rows,
        "output": str(output_path),
    }


def combine_split_csvs(output_dir: Path) -> Path:
    combined_path = output_dir / "kmhas_all_splits_clean.csv"
    split_paths = [
        output_dir / "kmhas_train_clean.csv",
        output_dir / "kmhas_valid_clean.csv",
        output_dir / "kmhas_test_clean.csv",
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


def zip_outputs(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/kmhas/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/kmhas/processed/clean"))
    parser.add_argument("--zip-path", type=Path, default=Path("data/kmhas/processed/kmhas_clean_standardized.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            raise FileNotFoundError(f"Missing raw split: {input_path}")
        summary.append(process_split(input_path, args.output_dir / f"kmhas_{split}_clean.csv", split))

    combine_split_csvs(args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "adlnlp/K-MHaS",
                "perturbation_type": "clean",
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# K-MHaS Clean Standardized Dataset\n\n"
        "Clean K-MHaS splits converted to the same CSV schema used by glyph-mix datasets.\n",
        encoding="utf-8",
    )
    zip_outputs(args.output_dir, args.zip_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"zip={args.zip_path}")


if __name__ == "__main__":
    main()

