#!/usr/bin/env python3
"""Train KoELECTRA on clean K-MHaS and evaluate the aligned benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ROOT,
    ensure_output_dirs,
    metrics_by_variant,
)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def balanced_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    per_class = rows // 2
    hate = frame[frame["is_hate_speech"] == 1].sample(n=per_class, random_state=seed)
    non_hate = frame[frame["is_hate_speech"] == 0].sample(
        n=rows - per_class, random_state=seed
    )
    return (
        pd.concat([hate, non_hate], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="monologg/koelectra-small-v3-discriminator")
    parser.add_argument("--train-rows", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models/kmhas_free_llm_benchmark/koelectra_small",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DATA_ROOT / "benchmark_test_2500.csv",
    )
    parser.add_argument("--reuse-saved", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()

    if args.reuse_saved and (args.model_dir / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
        trained = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ).to(device)
        clean_train = pd.read_csv(
            ROOT / "data/kmhas/processed/clean/kmhas_train_clean.csv"
        )
        train = balanced_sample(clean_train, args.train_rows, args.seed)
        dataset = TextDataset(
            train["document"].fillna("").astype(str).tolist(),
            train["is_hate_speech"].astype(int).to_numpy(),
            tokenizer,
            args.max_len,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for step, batch in enumerate(loader, start=1):
                optimizer.zero_grad()
                loss = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                ).loss
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if step % 50 == 0:
                    print(
                        f"epoch={epoch} step={step}/{len(loader)} "
                        f"recent_loss={np.mean(losses[-50:]):.4f}"
                    )
            print(f"epoch={epoch} mean_loss={np.mean(losses):.4f}")
        args.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        trained = True

    benchmark = pd.read_csv(args.benchmark)
    dataset = TextDataset(
        benchmark["text"].fillna("").astype(str).tolist(),
        benchmark["label"].astype(int).to_numpy(),
        tokenizer,
        args.max_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    probabilities = predict(model, loader, device)
    predictions = benchmark[
        ["split", "id", "variant", "label", "text", "normalized_text"]
    ].copy()
    predictions["system"] = "koelectra_small"
    predictions["score"] = probabilities
    predictions["threshold"] = 0.5
    predictions["pred_label"] = (probabilities >= 0.5).astype(int)
    predictions["raw_output"] = predictions["pred_label"].map(
        {0: "non-hate", 1: "hate"}
    )
    predictions.to_csv(PREDICTIONS_ROOT / "koelectra_small.csv", index=False)
    metrics = metrics_by_variant(predictions, "koelectra_small")
    metrics.to_csv(REPORTS_ROOT / "koelectra_small_metrics.csv", index=False)
    metadata = {
        **vars(args),
        "model_dir": str(args.model_dir),
        "benchmark": str(args.benchmark),
        "device": str(device),
        "trained_this_run": trained,
    }
    (REPORTS_ROOT / "koelectra_small_run_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(metrics[["variant", "balanced_accuracy", "macro_f1", "fnr", "fpr"]])


if __name__ == "__main__":
    main()
