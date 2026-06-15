#!/usr/bin/env python3
"""Aggregate benchmark metrics, render figures, and write the Korean report."""

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
    FIGURES_ROOT,
    PREDICTIONS_ROOT,
    REPORTS_ROOT,
    ROOT,
    VARIANT_LABELS,
    VARIANT_ORDER,
    ensure_output_dirs,
    summarize_systems,
)


DISPLAY_NAMES = {
    "char_tfidf_mlp": "Char TF-IDF MLP",
    "normalization_char_tfidf_mlp": "Normalization + Char MLP",
    "koelectra_small": "KoELECTRA-small",
    "qwen_raw_uncalibrated": "Qwen raw zero-shot",
    "qwen_raw_calibrated": "Qwen calibrated",
    "qwen_normalized_uncalibrated": "Qwen normalized",
    "qwen_normalized_calibrated": "Qwen normalized + calibrated",
    "qwen_lora_normalized_uncalibrated": "Qwen LoRA + normalized",
    "qwen_lora_normalized_calibrated": "Qwen final",
    "gemini_2_5_flash_lite": "Gemini 2.5 Flash-Lite",
}

PLOT_SYSTEM_ORDER = [
    "char_tfidf_mlp",
    "normalization_char_tfidf_mlp",
    "koelectra_small",
    "qwen_raw_uncalibrated",
    "qwen_raw_calibrated",
    "qwen_normalized_calibrated",
    "qwen_lora_normalized_calibrated",
    "gemini_2_5_flash_lite",
]

RANKING_SYSTEM_ORDER = [
    "char_tfidf_mlp",
    "normalization_char_tfidf_mlp",
    "koelectra_small",
    "qwen_raw_uncalibrated",
    "qwen_lora_normalized_calibrated",
    "gemini_2_5_flash_lite",
]

QWEN_STAGE_ORDER = [
    "qwen_raw_uncalibrated",
    "qwen_raw_calibrated",
    "qwen_normalized_calibrated",
    "qwen_lora_normalized_calibrated",
]

UI_PREFIXES = ("chatgpt", "claude", "gemini_free", "grok_free")

AGGREGATE_METRIC_FILES = {
    "all_model_metrics.csv",
    "supplementary_ui_metrics.csv",
    "external_ui_250_variant_metrics.csv",
    "external_ui_250_overall_metrics.csv",
}


def load_metrics() -> pd.DataFrame:
    frames = []
    for path in sorted(REPORTS_ROOT.glob("*_metrics.csv")):
        if path.name in AGGREGATE_METRIC_FILES:
            continue
        frame = pd.read_csv(path)
        if {"system", "variant", "balanced_accuracy"}.issubset(frame.columns):
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        raise RuntimeError("No benchmark metric files were found")
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.drop_duplicates(["system", "split", "variant"], keep="last")
    return metrics


def available_order(metrics: pd.DataFrame, requested: list[str]) -> list[str]:
    available = set(metrics["system"])
    return [system for system in requested if system in available]


def display(system: str) -> str:
    return DISPLAY_NAMES.get(system, system.replace("_", " "))


def style_axes(ax, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_balanced_accuracy(metrics: pd.DataFrame, systems: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(VARIANT_ORDER))
    for system in systems:
        sub = metrics[metrics["system"] == system].set_index("variant").reindex(VARIANT_ORDER)
        ax.plot(
            x,
            sub["balanced_accuracy"],
            marker="o",
            linewidth=2.2,
            label=display(system),
        )
    ax.set_xticks(x, [VARIANT_LABELS[v] for v in VARIANT_ORDER], rotation=12)
    ax.set_ylim(0.45, 1.0)
    style_axes(ax, "Balanced accuracy", "Robustness across obfuscation variants")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "balanced_accuracy_by_variant.png", dpi=220)
    plt.close(fig)


def plot_dual_failure(metrics: pd.DataFrame, systems: list[str]) -> None:
    sub = metrics[
        (metrics["variant"] == "roman_glyph_70") & metrics["system"].isin(systems)
    ].copy()
    sub["order"] = sub["system"].map({s: i for i, s in enumerate(systems)})
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - width / 2, sub["fnr"], width, label="FNR: hate missed", color="#D55E00")
    ax.bar(x + width / 2, sub["fpr"], width, label="FPR: non-hate blocked", color="#0072B2")
    ax.set_xticks(x, [display(s) for s in sub["system"]], rotation=22, ha="right")
    ax.set_ylim(0, 1)
    style_axes(ax, "Failure rate", "Dual failure on Roman+Glyph 70")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "fnr_fpr_dual_failure.png", dpi=220)
    plt.close(fig)


def plot_qwen_stages(metrics: pd.DataFrame) -> None:
    systems = available_order(metrics, QWEN_STAGE_ORDER)
    if not systems:
        return
    sub = metrics[
        (metrics["variant"] == "roman_glyph_70") & metrics["system"].isin(systems)
    ].copy()
    sub["order"] = sub["system"].map({s: i for i, s in enumerate(systems)})
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(x, sub["balanced_accuracy"], color=["#999999", "#E69F00", "#56B4E9", "#009E73"][: len(sub)])
    ax.set_xticks(x, [display(s) for s in sub["system"]], rotation=18, ha="right")
    ax.set_ylim(0.45, max(0.85, float(sub["balanced_accuracy"].max()) + 0.08))
    style_axes(ax, "Balanced accuracy", "Qwen improvement stages on Roman+Glyph 70")
    for bar, value in zip(bars, sub["balanced_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "qwen_improvement_stages.png", dpi=220)
    plt.close(fig)


def plot_predicted_hate_rate(metrics: pd.DataFrame, systems: list[str]) -> None:
    sub = metrics[
        (metrics["variant"] == "roman_glyph_70") & metrics["system"].isin(systems)
    ].copy()
    sub["order"] = sub["system"].map({s: i for i, s in enumerate(systems)})
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    bars = ax.bar(x, sub["predicted_hate_rate"], color="#CC79A7")
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.5, label="True hate rate = 0.50")
    ax.set_xticks(x, [display(s) for s in sub["system"]], rotation=22, ha="right")
    ax.set_ylim(0, 1)
    style_axes(ax, "Predicted hate rate", "Over-blocking check on Roman+Glyph 70")
    ax.legend(frameon=False)
    for bar, value in zip(bars, sub["predicted_hate_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "predicted_hate_rate.png", dpi=220)
    plt.close(fig)


def plot_confusion_matrices() -> None:
    candidates = [
        ("qwen_raw_uncalibrated.csv", "Raw Qwen"),
        ("qwen_lora_normalized_calibrated.csv", "Final Qwen"),
        ("qwen_normalized_calibrated.csv", "Normalized Qwen"),
    ]
    selected: list[tuple[Path, str]] = []
    for filename, title in candidates:
        path = PREDICTIONS_ROOT / filename
        if path.exists() and all(path != prior for prior, _ in selected):
            selected.append((path, title))
        if len(selected) == 2:
            break
    if len(selected) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, (path, title) in zip(axes, selected):
        frame = pd.read_csv(path)
        frame = frame[frame["variant"] == "roman_glyph_70"]
        tn = int(((frame["label"] == 0) & (frame["pred_label"] == 0)).sum())
        fp = int(((frame["label"] == 0) & (frame["pred_label"] == 1)).sum())
        fn = int(((frame["label"] == 1) & (frame["pred_label"] == 0)).sum())
        tp = int(((frame["label"] == 1) & (frame["pred_label"] == 1)).sum())
        matrix = np.array([[tn, fp], [fn, tp]])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=250)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred non-hate", "Pred hate"])
        ax.set_yticks([0, 1], ["True non-hate", "True hate"])
        ax.set_title(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    fig.suptitle("Qwen confusion matrices on Roman+Glyph 70", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.14, top=0.82, wspace=0.35)
    fig.savefig(FIGURES_ROOT / "qwen_confusion_matrices.png", dpi=220)
    plt.close(fig)


def plot_ranking(summary: pd.DataFrame) -> None:
    ranked = summary.sort_values("worst_case_balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(ranked) * 0.6)))
    bars = ax.barh(
        [display(s) for s in ranked["system"]],
        ranked["worst_case_balanced_accuracy"],
        color="#009E73",
    )
    ax.set_xlim(0.45, 1.0)
    style_axes(ax, "Worst-case balanced accuracy", "Final robustness ranking")
    for bar, value in zip(bars, ranked["worst_case_balanced_accuracy"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "final_system_ranking.png", dpi=220)
    plt.close(fig)


def export_error_examples() -> None:
    requests = [
        ("char_tfidf_mlp.csv", "roman_glyph_70", 1, 0, "traditional_under_detection"),
        ("qwen_raw_uncalibrated.csv", "roman_glyph_70", 0, 1, "qwen_over_blocking"),
        (
            "normalization_char_tfidf_mlp.csv",
            "roman_glyph_70",
            1,
            0,
            "normalizer_remaining_under_detection",
        ),
        (
            "qwen_lora_normalized_calibrated.csv",
            "roman_glyph_70",
            0,
            1,
            "final_qwen_remaining_over_blocking",
        ),
    ]
    examples = []
    for filename, variant, true_label, pred_label, error_type in requests:
        path = PREDICTIONS_ROOT / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        sub = frame[
            (frame["variant"] == variant)
            & (frame["label"] == true_label)
            & (frame["pred_label"] == pred_label)
        ].head(20)
        sub = sub.copy()
        sub["source_prediction_file"] = filename
        sub["error_type"] = error_type
        examples.append(sub)
    if examples:
        pd.concat(examples, ignore_index=True).to_csv(
            REPORTS_ROOT / "representative_error_examples.csv", index=False
        )


def fmt(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.3f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ui_metrics: pd.DataFrame,
) -> Path:
    ranking = summary.copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["display_name"] = ranking["system"].map(display)
    winner = ranking.iloc[0]
    strongest = metrics[metrics["variant"] == "roman_glyph_70"].copy()
    strongest["display_name"] = strongest["system"].map(display)
    strongest = strongest.sort_values("balanced_accuracy", ascending=False)

    expected = {
        "KoELECTRA-small": "koelectra_small",
        "Qwen final LoRA": "qwen_lora_normalized_calibrated",
        "Gemini API": "gemini_2_5_flash_lite",
    }
    missing = [label for label, system in expected.items() if system not in set(metrics["system"])]
    status = "없음" if not missing else ", ".join(missing)
    raw_qwen = metrics[
        (metrics["system"] == "qwen_raw_uncalibrated")
        & (metrics["variant"] == "roman_glyph_70")
    ]
    raw_qwen_text = "아직 평가되지 않음"
    if not raw_qwen.empty:
        row = raw_qwen.iloc[0]
        raw_qwen_text = (
            f"BA {row['balanced_accuracy']:.3f}, FNR {row['fnr']:.3f}, "
            f"FPR {row['fpr']:.3f}, predicted hate rate {row['predicted_hate_rate']:.3f}"
        )
    final_qwen = metrics[
        (metrics["system"] == "qwen_lora_normalized_calibrated")
        & (metrics["variant"] == "roman_glyph_70")
    ]
    if final_qwen.empty:
        qwen_improvement_text = "LoRA final checkpoint 평가는 아직 실행 대기 중입니다."
    else:
        final_row = final_qwen.iloc[0]
        raw_row = raw_qwen.iloc[0]
        qwen_improvement_text = (
            f"Roman+Glyph 70에서 balanced accuracy가 "
            f"`{raw_row['balanced_accuracy']:.3f}`에서 `{final_row['balanced_accuracy']:.3f}`로, "
            f"FPR이 `{raw_row['fpr']:.3f}`에서 `{final_row['fpr']:.3f}`로 변했습니다."
        )

    report = f"""# 무료 LLM 비교 및 Qwen 개선 실험 결과

> K-MHaS의 동일한 test ID 500개를 다섯 난독화 variant로 변환해 전통 분류기와 로컬 sLLM을 비교하고, Qwen의 과잉 차단을 normalization, threshold calibration, LoRA로 개선하는 robustness 실험입니다.

## 1. 실험 상태

- 공통 test: 500개 base comment, hate/non-hate 250개씩
- 총 평가 입력: 모델당 2,500개
- Variant: `clean`, `romanized_35`, `romanized_70`, `roman_glyph_35`, `roman_glyph_70`
- 주 순위 지표: worst-case balanced accuracy
- 아직 외부 실행이 필요한 항목: {status}

## 2. 데이터와 누수 방지

- K-MHaS의 train, validation, test split을 그대로 분리했습니다.
- LoRA train ID, LoRA validation ID, threshold calibration ID, final test ID는 서로 겹치지 않습니다.
- `romanized_70`과 `roman_glyph_70`은 LoRA 학습에 넣지 않고 강한 난독화 generalization 평가에만 사용했습니다.
- 모든 test variant는 동일한 base ID 500개와 동일한 binary label을 공유합니다.

## 3. 비교 시스템

- `char_tfidf_mlp`: 문자 n-gram 전통 baseline
- `normalization + char_mlp`: romanized/glyph 입력을 규칙 기반으로 복원한 뒤 분류
- `KoELECTRA-small`: 한국어 Transformer reference
- `Qwen raw zero-shot`: 난독화 원문을 그대로 logit scoring
- `Qwen calibrated`: validation에서 hate/non-hate decision threshold 선택
- `Qwen normalization`: 원문과 규칙 기반 복원 후보를 함께 제공
- `Qwen final`: normalization + LoRA + calibrated threshold
- Gemini와 ChatGPT/Claude UI는 무료 접근이 가능한 경우 별도 실행

## 4. Qwen 개선 과정

Raw Qwen은 강한 난독화에서 **{raw_qwen_text}**를 기록했습니다. FNR만 보면 낮지만 non-hate까지 hate로 판단하는 FPR이 높아, 단순히 “악성을 잘 잡는다”고 해석할 수 없습니다.

1. Raw zero-shot으로 과잉 hate 편향을 재현했습니다.
2. 생성 문장을 파싱하지 않고 `1`과 `0` token logit 차이를 사용했습니다.
3. 독립 calibration set에서 `(FNR + FPR) / 2`가 최소인 threshold를 고정했습니다.
4. 원문과 normalized candidate를 함께 제공하는 ablation을 수행했습니다.
5. clean, romanized 35, roman+glyph 35 총 12,000개로 LoRA를 학습했습니다.
6. 1,500, 3,000, 4,500 step checkpoint는 calibration 성능으로 선택하고 test는 best checkpoint에 한 번만 사용합니다.

{qwen_improvement_text}

### Training Safety Check

초기 LoRA run에서는 긴 문장이 256 token에서 잘릴 때 끝의 assistant label까지 사라져 zero-supervision batch와 NaN loss가 발생했습니다. 해당 adapter는 결과에서 제외했습니다. 최종 데이터는 Qwen tokenizer 기준 head+tail truncation을 적용해 모든 13,500개 train/validation 예시가 256 token 이하이고 supervised token이 3개 존재하는지 검사했습니다.

## 5. 핵심 결론

현재 완료된 시스템 중 **best tested system은 `{display(winner['system'])}`**입니다.

- Worst-case balanced accuracy: **{winner['worst_case_balanced_accuracy']:.3f}**
- 전체 variant 평균 balanced accuracy: **{winner['mean_balanced_accuracy']:.3f}**
- Worst max(FNR, FPR): **{winner['worst_max_failure_rate']:.3f}**

이 순위는 clean 성능 하나가 아니라, 다섯 variant 중 가장 낮은 balanced accuracy를 먼저 비교하고 실패율과 평균 성능을 tie-breaker로 사용한 결과입니다.

## 6. 최종 순위

{markdown_table(
    ranking,
    ["rank", "display_name", "worst_case_balanced_accuracy", "worst_max_failure_rate", "mean_balanced_accuracy"],
    ["Rank", "System", "Worst-case BA", "Worst max failure", "Mean BA"],
)}

## 7. 가장 강한 난독화 결과

{markdown_table(
    strongest,
    ["display_name", "balanced_accuracy", "macro_f1", "fnr", "fpr", "predicted_hate_rate"],
    ["System", "Balanced Acc.", "Macro F1", "FNR", "FPR", "Pred. hate rate"],
)}

## 8. 그래프

### 난독화 강도별 Balanced Accuracy

![Balanced accuracy](outputs/kmhas_free_llm_benchmark/figures/balanced_accuracy_by_variant.png)

### Under-detection과 Over-blocking

![FNR and FPR](outputs/kmhas_free_llm_benchmark/figures/fnr_fpr_dual_failure.png)

### Qwen 개선 단계

![Qwen stages](outputs/kmhas_free_llm_benchmark/figures/qwen_improvement_stages.png)

### Predicted Hate Rate

![Predicted hate rate](outputs/kmhas_free_llm_benchmark/figures/predicted_hate_rate.png)

### Qwen Confusion Matrix

![Qwen confusion matrices](outputs/kmhas_free_llm_benchmark/figures/qwen_confusion_matrices.png)

### 최종 Ranking

![Final ranking](outputs/kmhas_free_llm_benchmark/figures/final_system_ranking.png)

## 9. 해석 기준

- `FNR`: 실제 hate를 non-hate로 놓친 비율입니다. 낮을수록 좋습니다.
- `FPR`: 실제 non-hate를 hate로 막은 비율입니다. 낮을수록 좋습니다.
- `Predicted Hate Rate`: benchmark의 실제 hate 비율은 50%입니다. 이 값이 지나치게 높으면 과잉 차단 가능성이 큽니다.
- `Worst-case Balanced Accuracy`: 다섯 variant 중 가장 낮은 balanced accuracy입니다. 난독화에 가장 취약한 순간을 대표합니다.

## 10. 사회적 의미

- FNR이 높은 모델은 실제 악성 댓글을 놓쳐 피해 노출을 늘릴 수 있습니다.
- FPR이 높은 모델은 정상적인 비판과 일반 댓글까지 차단해 표현의 자유와 서비스 신뢰를 해칠 수 있습니다.
- 따라서 moderation 시스템은 탐지율 하나가 아니라 under-detection과 over-blocking을 함께 관리해야 합니다.
- 본 프로젝트의 “best”는 절대적인 최고 모델이 아니라, 정의한 K-MHaS 난독화 threat model에서 두 오류가 가장 균형 잡힌 시스템을 뜻합니다.

## 11. 재현 파일

- Benchmark: `outputs/kmhas_free_llm_benchmark/data/benchmark_test_2500.csv`
- ID manifest: `outputs/kmhas_free_llm_benchmark/data/benchmark_test_ids.csv`
- Predictions: `outputs/kmhas_free_llm_benchmark/predictions/`
- Metrics: `outputs/kmhas_free_llm_benchmark/reports/all_model_metrics.csv`
- Ranking: `outputs/kmhas_free_llm_benchmark/reports/main_system_ranking.csv`
- LoRA recipe: `configs/qwen_lora_mlx.yaml`

## 12. UI 보조 비교

ChatGPT Free/Plus와 Claude Free 결과는 backend와 rate limit을 통제할 수 없으므로 메인 순위에 합치지 않습니다. 동일한 250개 UI subset과 prompt를 사용한 참고 결과만 별도로 기록합니다.

UI 결과 행 수: **{len(ui_metrics)}**
"""
    path = ROOT / "notion_free_llm_qwen_improvement_results_ko.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    metrics.to_csv(REPORTS_ROOT / "all_model_metrics.csv", index=False)

    ui_mask = metrics["system"].astype(str).str.startswith(UI_PREFIXES)
    ui_metrics = metrics[ui_mask].copy()
    main_metrics = metrics[~ui_mask].copy()
    systems = available_order(main_metrics, PLOT_SYSTEM_ORDER)
    main_metrics = main_metrics[main_metrics["system"].isin(systems)].copy()
    ranking_systems = available_order(main_metrics, RANKING_SYSTEM_ORDER)
    ranking_metrics = main_metrics[main_metrics["system"].isin(ranking_systems)].copy()
    summary = summarize_systems(ranking_metrics)
    summary.to_csv(REPORTS_ROOT / "main_system_ranking.csv", index=False)
    best = summary.iloc[[0]].copy()
    best.insert(0, "conclusion", "best tested system on this K-MHaS obfuscation benchmark")
    best.to_csv(REPORTS_ROOT / "best_tested_system.csv", index=False)
    presentation_table = main_metrics[
        main_metrics["variant"] == "roman_glyph_70"
    ][
        [
            "system",
            "balanced_accuracy",
            "macro_f1",
            "fnr",
            "fpr",
            "predicted_hate_rate",
        ]
    ].sort_values("balanced_accuracy", ascending=False)
    presentation_table.to_csv(
        REPORTS_ROOT / "presentation_roman_glyph70_key_numbers.csv", index=False
    )
    qwen_stage_table = main_metrics[
        (main_metrics["variant"] == "roman_glyph_70")
        & (main_metrics["system"].isin(QWEN_STAGE_ORDER))
    ][
        [
            "system",
            "balanced_accuracy",
            "macro_f1",
            "fnr",
            "fpr",
            "predicted_hate_rate",
        ]
    ]
    qwen_stage_table.to_csv(
        REPORTS_ROOT / "qwen_improvement_key_numbers.csv", index=False
    )
    ui_metrics.to_csv(REPORTS_ROOT / "supplementary_ui_metrics.csv", index=False)

    plot_balanced_accuracy(main_metrics, systems)
    plot_dual_failure(main_metrics, systems)
    plot_qwen_stages(main_metrics)
    plot_predicted_hate_rate(main_metrics, systems)
    plot_confusion_matrices()
    plot_ranking(summary)
    export_error_examples()
    report_path = write_report(main_metrics, summary, ui_metrics)

    result = {
        "systems": systems,
        "best_tested_system": summary.iloc[0]["system"],
        "report": str(report_path.relative_to(ROOT)),
        "figures": sorted(path.name for path in FIGURES_ROOT.glob("*.png")),
    }
    (REPORTS_ROOT / "summary_run_info.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
