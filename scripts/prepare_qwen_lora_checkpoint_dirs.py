#!/usr/bin/env python3
"""Create MLX-LM adapter directories for epoch checkpoint evaluation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("models/kmhas_free_llm_benchmark/qwen_lora"),
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1500, 3000, 4500])
    args = parser.parse_args()

    config = args.adapter_root / "adapter_config.json"
    if not config.exists():
        raise FileNotFoundError(config)

    for step in args.steps:
        source = args.adapter_root / f"{step:07d}_adapters.safetensors"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = args.adapter_root / f"checkpoint-{step}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, destination / "adapter_config.json")
        shutil.copy2(source, destination / "adapters.safetensors")
        print(destination)


if __name__ == "__main__":
    main()
