# K-MHaS 무료 LLM Robustness Benchmark

## 목적

동일한 한국어 댓글을 난독화 강도별로 변형했을 때 전통 분류기와 sLLM이 보이는 두 실패를 함께 측정합니다.

- Under-detection: 실제 hate를 non-hate로 놓치는 문제
- Over-blocking: 실제 non-hate를 hate로 차단하는 문제

최종 시스템은 clean F1이 아니라 `worst-case balanced accuracy`를 우선해 선정합니다.

## 최종 결과

현재 로컬에서 재현 완료한 시스템 중 최종 순위는 다음과 같습니다.

| Rank | System | Worst-case BA | Mean BA | Worst max(FNR, FPR) |
|---|---|---:|---:|---:|
| 1 | Normalization + Char TF-IDF MLP | 0.772 | 0.797 | 0.312 |
| 2 | Qwen LoRA + normalization + calibration | 0.672 | 0.716 | 0.396 |
| 3 | Char TF-IDF MLP | 0.572 | 0.699 | 0.848 |
| 4 | KoELECTRA-small | 0.544 | 0.659 | 0.880 |
| 5 | Qwen raw zero-shot | 0.522 | 0.562 | 0.852 |

가장 강한 `roman_glyph_70`에서 Qwen은 다음과 같이 개선됐습니다.

- Raw zero-shot: BA `0.522`, FNR `0.104`, FPR `0.852`
- Final Qwen: BA `0.672`, FNR `0.268`, FPR `0.388`

즉 LoRA와 calibration은 Qwen의 과잉 hate 판단을 크게 줄였지만, 이번 benchmark의 최종 best tested system은 규칙 기반 복원과 Char MLP를 결합한 시스템입니다.

## 공통 Benchmark

- Base test comment: 500개
- Hate / non-hate: 250 / 250
- Variant: 5개
- 모델당 prediction: 2,500개
- Test 파일: `outputs/kmhas_free_llm_benchmark/data/benchmark_test_2500.csv`
- Calibration 파일: `outputs/kmhas_free_llm_benchmark/data/benchmark_calibration_2500.csv`
- Split audit: `outputs/kmhas_free_llm_benchmark/data/benchmark_metadata.json`

Variant:

1. `clean`
2. `romanized_35`
3. `romanized_70`
4. `roman_glyph_35`
5. `roman_glyph_70`

## 모델

- `char_tfidf_mlp`
- `normalization_char_tfidf_mlp`
- `KoELECTRA-small`
- `Qwen2.5-1.5B raw zero-shot`
- `Qwen2.5-1.5B calibrated`
- `Qwen2.5-1.5B normalization + calibrated`
- `Qwen2.5-1.5B LoRA + normalization + calibrated`
- `Gemini 2.5 Flash-Lite` 무료 API
- ChatGPT Free/Plus, Claude Free UI 보조 평가

## Qwen LoRA Recipe

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime: MLX-LM, Apple Silicon
- Train base ID: 4,000개
- Train variant: `clean`, `romanized_35`, `roman_glyph_35`
- 총 train example: 12,000개
- Hate/non-hate: 50:50
- Rank: 8
- Alpha equivalent: 16 (`scale=2.0`)
- Initial trial learning rate: `1e-4`
- 안정화 learning rate: `2e-5`, 100-step warmup, cosine decay
- AdamW: `eps=1e-6`, `weight_decay=0`, bias correction 사용
- Batch size: 8
- Gradient accumulation: 1
- Max length: 256
- Token-aware head+tail truncation으로 assistant 정답 토큰이 항상 256 token 안에 포함되도록 보장
- Epoch: 3
- Checkpoint: 1,500 / 3,000 / 4,500 step
- Best checkpoint: 4,500 step
- Best-checkpoint calibration BA: `0.7364`
- 강한 `romanized_70`, `roman_glyph_70`은 학습에서 제외
- Config: `configs/qwen_lora_mlx.yaml`

초기 run은 긴 입력이 256 token에서 잘릴 때 문장 끝의 assistant 정답 토큰까지 사라져 supervised token 수가 0이 되는 문제로 loss가 `NaN`이 되었습니다. 해당 checkpoint는 폐기했습니다. 최종 run은 token-aware truncation으로 정답 보존을 검증하고, learning rate와 AdamW epsilon도 보수적으로 낮춘 안정화 설정만 사용합니다.

## 재현 순서

```bash
.venv/bin/python scripts/prepare_kmhas_free_llm_benchmark.py
.venv/bin/python scripts/evaluate_saved_classifiers_free_llm_benchmark.py
.venv/bin/python scripts/train_evaluate_koelectra_free_llm_benchmark.py

.venv/bin/python scripts/evaluate_qwen_mlx_free_llm_benchmark.py \
  --input-mode raw --run-name qwen_raw --resume

.venv/bin/python scripts/evaluate_qwen_mlx_free_llm_benchmark.py \
  --input-mode normalized --run-name qwen_normalized --resume

.venv/bin/mlx_lm.lora --config configs/qwen_lora_mlx.yaml
.venv/bin/python scripts/prepare_qwen_lora_checkpoint_dirs.py

for step in 1500 3000 4500; do
  .venv/bin/python scripts/evaluate_qwen_mlx_free_llm_benchmark.py \
    --adapter-path "models/kmhas_free_llm_benchmark/qwen_lora/checkpoint-${step}" \
    --input-mode normalized \
    --run-name "qwen_lora_step${step}" \
    --calibration-only --resume
done

.venv/bin/python scripts/select_best_qwen_lora_checkpoint.py

.venv/bin/python scripts/evaluate_qwen_mlx_free_llm_benchmark.py \
  --adapter-path models/kmhas_free_llm_benchmark/qwen_lora/best \
  --input-mode normalized \
  --run-name qwen_lora_normalized --resume
```

Gemini API key가 설정된 경우:

```bash
GEMINI_API_KEY=... .venv/bin/python scripts/evaluate_gemini_free_llm_benchmark.py
```

최종 집계:

```bash
MPLCONFIGDIR=/tmp/matplotlib-safe-ai \
XDG_CACHE_HOME=/tmp/xdg-cache-safe-ai \
.venv/bin/python scripts/summarize_free_llm_benchmark.py
```

## ChatGPT / Claude UI 보조 평가

`outputs/kmhas_free_llm_benchmark/data/ui_batches/README.md`의 prompt와 JSONL batch를 사용합니다.

결과 CSV 형식:

```csv
id,variant,label
sample-id,clean,hate
```

가져오기:

```bash
.venv/bin/python scripts/import_ui_llm_predictions.py \
  --input result.csv \
  --system chatgpt_free \
  --displayed-model "UI에 표시된 모델명" \
  --run-date 2026-06-09
```

UI 결과는 정확한 backend와 rate limit을 통제할 수 없으므로 메인 ranking에 포함하지 않습니다.

## 주요 산출물

- 모델 prediction: `outputs/kmhas_free_llm_benchmark/predictions/`
- 통합 metrics: `outputs/kmhas_free_llm_benchmark/reports/all_model_metrics.csv`
- 최종 ranking: `outputs/kmhas_free_llm_benchmark/reports/main_system_ranking.csv`
- Figure: `outputs/kmhas_free_llm_benchmark/figures/`
- 노션 문서: `notion_free_llm_qwen_improvement_results_ko.md`
- Qwen adapter: `models/kmhas_free_llm_benchmark/qwen_lora/best/`
- Benchmark integrity audit: `outputs/kmhas_free_llm_benchmark/reports/benchmark_integrity_report.json`
- 공유 패키지: `kmhas_free_llm_benchmark_share_20260611.zip`

## 비용

- Qwen 및 KoELECTRA: 로컬 실행
- Gemini: 무료 API tier만 사용
- ChatGPT / Claude: 무료 UI 또는 현재 계정 UI 보조 평가
- 유료 API 호출과 실제 서비스 댓글 게시는 사용하지 않음

## 외부 무료 UI 보조 결과

2026년 6월 13일, 동일한 base comment 50개와 5개 variant로 구성한
250행 표본을 ChatGPT, Gemini, Claude, Grok 무료 UI에서 평가했습니다.

| System | Worst-case BA | Mean BA |
|---|---:|---:|
| Gemini Free UI | 0.860 | 0.892 |
| Claude Free UI | 0.800 | 0.832 |
| ChatGPT Free UI | 0.780 | 0.820 |
| Grok Free UI | 0.700 | 0.796 |

정확한 backend 모델명이 기록되지 않았고 표본도 작으므로 이 결과는
2,500행 로컬 main leaderboard와 분리한 supplementary comparison입니다.
상세 내용은
`outputs/kmhas_free_llm_benchmark/reports/external_ui_250_comparison_ko.md`에
있습니다.
