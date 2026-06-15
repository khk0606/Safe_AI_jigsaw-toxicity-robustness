#!/usr/bin/env python3
"""Compare returned free-UI LLM results with local systems on the same 250 rows."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-safe-ai")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-safe-ai")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmhas_free_llm_benchmark_common import (
    DATA_ROOT,
    FIGURES_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ROOT,
    VARIANT_LABELS,
    VARIANT_ORDER,
    metrics_by_variant,
    prediction_metrics,
    summarize_systems,
)


SYSTEMS = [
    "normalization_char_tfidf_mlp",
    "qwen_lora_normalized_calibrated",
    "qwen_raw_uncalibrated",
    "chatgpt_free",
    "gemini_free",
    "claude_free",
    "grok_free",
]

DISPLAY_NAMES = {
    "normalization_char_tfidf_mlp": "Normalization + Char MLP",
    "qwen_lora_normalized_calibrated": "Qwen final",
    "qwen_raw_uncalibrated": "Qwen raw",
    "chatgpt_free": "ChatGPT Free UI",
    "gemini_free": "Gemini Free UI",
    "claude_free": "Claude Free UI",
    "grok_free": "Grok Free UI",
}

COLORS = {
    "normalization_char_tfidf_mlp": "#157F6B",
    "qwen_lora_normalized_calibrated": "#2B6CB0",
    "qwen_raw_uncalibrated": "#8795A1",
    "chatgpt_free": "#10A37F",
    "gemini_free": "#4285F4",
    "claude_free": "#C15F3C",
    "grok_free": "#202124",
}


def load_aligned_predictions(system: str, manifest: pd.DataFrame) -> pd.DataFrame:
    path = PREDICTIONS_ROOT / f"{system}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    predictions = pd.read_csv(path)
    predictions["id"] = predictions["id"].astype(str)
    pairs = manifest[["id", "variant"]].copy()
    pairs["id"] = pairs["id"].astype(str)
    selected = pairs.merge(
        predictions,
        on=["id", "variant"],
        how="left",
        validate="one_to_one",
    )
    if selected["pred_label"].isna().any():
        raise ValueError(
            f"{system} is missing {int(selected['pred_label'].isna().sum())} predictions"
        )
    selected["split"] = "test"
    selected["label"] = manifest["label"].astype(int).to_numpy()
    selected["system"] = system
    return selected


def overall_metrics(predictions: pd.DataFrame, system: str) -> dict:
    metrics = prediction_metrics(predictions)
    metrics["system"] = system
    return metrics


def plot_variant_metrics(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(VARIANT_ORDER))
    for system in SYSTEMS:
        group = metrics[metrics["system"] == system].set_index("variant")
        values = [group.loc[variant, "balanced_accuracy"] for variant in VARIANT_ORDER]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            label=DISPLAY_NAMES[system],
            color=COLORS[system],
        )
    ax.set_xticks(x, [VARIANT_LABELS[v] for v in VARIANT_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Free UI LLM comparison on the same 250-row subset", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dual_failure(metrics: pd.DataFrame, path: Path) -> None:
    strongest = metrics[metrics["variant"] == "roman_glyph_70"].copy()
    strongest["display"] = strongest["system"].map(DISPLAY_NAMES)
    strongest = strongest.sort_values("balanced_accuracy", ascending=False)
    x = np.arange(len(strongest))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - width / 2, strongest["fnr"], width, label="FNR: hate missed", color="#E45756")
    ax.bar(x + width / 2, strongest["fpr"], width, label="FPR: non-hate blocked", color="#4C78A8")
    ax.set_xticks(x, strongest["display"], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Failure rate")
    ax.set_title("Dual failure on Roman+Glyph 70", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    manifest = pd.read_csv(DATA_ROOT / "ui_supplementary_250.csv")
    all_predictions = []
    metric_frames = []
    overall_rows = []

    for system in SYSTEMS:
        predictions = load_aligned_predictions(system, manifest)
        all_predictions.append(predictions)
        metric_frames.append(metrics_by_variant(predictions, system))
        overall_rows.append(overall_metrics(predictions, system))

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_systems(metrics)
    overall = pd.DataFrame(overall_rows)
    for frame in (metrics, summary, overall):
        frame["display_name"] = frame["system"].map(DISPLAY_NAMES)

    metrics.to_csv(REPORTS_ROOT / "external_ui_250_variant_metrics.csv", index=False)
    summary.to_csv(REPORTS_ROOT / "external_ui_250_ranking.csv", index=False)
    overall.to_csv(REPORTS_ROOT / "external_ui_250_overall_metrics.csv", index=False)

    plot_variant_metrics(metrics, FIGURES_ROOT / "external_ui_250_balanced_accuracy.png")
    plot_dual_failure(metrics, FIGURES_ROOT / "external_ui_250_dual_failure.png")

    strongest = metrics[metrics["variant"] == "roman_glyph_70"].copy()
    strongest = strongest.sort_values("balanced_accuracy", ascending=False)
    ranking = summary.copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    report = f"""# 외부 무료 LLM 250개 공통 표본 비교 결과

## 실험 조건

- 실행일: 2026-06-13
- 동일 base comment: 50개
- Variant: 5개
- 총 입력: 시스템당 250개
- 각 variant: hate 25개, non-hate 25개
- UI 결과 형식 검증: 네 서비스 모두 250/250, 누락·중복·parse failure 0건

## 중요한 제한

- 팀원의 `RUN_LOG.csv`에 화면 표시 모델명이 기록되지 않았습니다.
- 따라서 결과는 특정 GPT/Claude/Gemini/Grok 모델 버전이 아니라 **2026-06-13 무료 UI 서비스 결과**입니다.
- 무료 UI는 backend 변경, fallback, rate limit을 통제할 수 없어 로컬 2,500개 실험과 분리한 supplementary comparison입니다.
- variant당 50개뿐이므로 Balanced Accuracy가 0.02 단위로 변합니다.

## Worst-case Robustness Ranking

{markdown_table(
    ranking,
    ["rank", "display_name", "worst_case_balanced_accuracy", "worst_max_failure_rate", "mean_balanced_accuracy"],
    ["Rank", "System", "Worst BA", "Worst max failure", "Mean BA"],
)}

## Roman+Glyph 70

{markdown_table(
    strongest,
    ["display_name", "balanced_accuracy", "macro_f1", "fnr", "fpr", "predicted_hate_rate"],
    ["System", "BA", "Macro F1", "FNR", "FPR", "Pred. hate rate"],
)}

## 핵심 해석

1. 무료 UI 모델 중 Gemini가 worst-case BA와 평균 BA 모두 가장 높았습니다.
2. ChatGPT는 FPR이 낮은 대신 clean FNR이 0.40으로 악성 댓글을 보수적으로 적게 잡았습니다.
3. Claude는 hate recall은 높지만 강한 romanization에서 FPR이 0.36까지 증가했습니다.
4. Grok은 Roman+Glyph 70에서 FPR 0.48로 정상 댓글 과잉 차단이 가장 크게 나타났습니다.
5. 직접 학습한 Qwen final은 raw Qwen보다 개선됐지만, 이 250개 표본에서는 무료 UI LLM보다 낮았습니다.
6. Normalization + Char MLP는 전체 2,500개 실험에서는 최종 1위였지만, 이 작은 250개 표본의 외부 LLM 비교에서는 순위가 달라질 수 있습니다.

## Figures

![Balanced accuracy](../figures/external_ui_250_balanced_accuracy.png)

![Dual failure](../figures/external_ui_250_dual_failure.png)
"""
    report_path = REPORTS_ROOT / "external_ui_250_comparison_ko.md"
    report_path.write_text(report, encoding="utf-8")

    metadata = {
        "run_date": "2026-06-13",
        "rows_per_system": 250,
        "unique_base_ids": int(manifest["id"].nunique()),
        "displayed_model_recorded": False,
        "systems": SYSTEMS,
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORTS_ROOT / "external_ui_250_comparison_run_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(ranking[
        [
            "rank",
            "display_name",
            "worst_case_balanced_accuracy",
            "worst_max_failure_rate",
            "mean_balanced_accuracy",
        ]
    ].to_string(index=False))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
