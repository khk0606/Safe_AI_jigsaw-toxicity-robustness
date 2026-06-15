#!/usr/bin/env python3
"""Select the LoRA epoch checkpoint using calibration balanced accuracy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from kmhas_free_llm_benchmark_common import REPORTS_ROOT, ROOT


def main() -> None:
    rows = []
    for path in sorted(REPORTS_ROOT.glob("qwen_lora_step*_calibration_selection.csv")):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one selection row in {path}")
        row = frame.iloc[0].to_dict()
        digits = "".join(character for character in str(row["run_name"]) if character.isdigit())
        row["step"] = int(digits)
        row["source_file"] = path.name
        rows.append(row)
    if not rows:
        raise RuntimeError("No LoRA checkpoint calibration results found")

    comparison = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "max_failure_rate", "macro_f1"],
        ascending=[False, True, False],
    )
    comparison.to_csv(REPORTS_ROOT / "qwen_lora_checkpoint_selection.csv", index=False)
    best_step = int(comparison.iloc[0]["step"])

    adapter_root = ROOT / "models/kmhas_free_llm_benchmark/qwen_lora"
    source = adapter_root / f"checkpoint-{best_step}"
    destination = adapter_root / "best"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "adapter_config.json", destination / "adapter_config.json")
    shutil.copy2(source / "adapters.safetensors", destination / "adapters.safetensors")

    result = {
        "best_step": best_step,
        "selection_metric": "calibration balanced_accuracy",
        "tie_breakers": ["lower max_failure_rate", "higher macro_f1"],
        "adapter_path": str(destination),
    }
    (REPORTS_ROOT / "qwen_lora_best_checkpoint.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
