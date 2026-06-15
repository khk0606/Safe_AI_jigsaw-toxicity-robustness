# Training Recipes and Results

## 1. Evaluation Protocol

Primary ranking metric:

```text
worst-case balanced accuracy across the five variants
```

Tie-breakers:

1. lower worst-case `max(FNR, FPR)`;
2. higher mean balanced accuracy.

Additional metrics:

- Macro F1
- FNR: hate comments incorrectly predicted as non-hate
- FPR: non-hate comments incorrectly predicted as hate
- predicted hate rate
- clean-to-perturbed performance drop

## 2. Character TF-IDF MLP

Prepare the clean and stress-test datasets:

```bash
python scripts/prepare_kmhas_clean_dataset.py

python scripts/create_kmhas_glyph_mix_dataset.py \
  --variant-name glyph_mix_35 \
  --syllable-ratio 0.35 \
  --glyph-ratio 0.35

python scripts/create_kmhas_glyph_mix_dataset.py \
  --variant-name glyph_mix_90 \
  --syllable-ratio 1.0 \
  --glyph-ratio 0.98

python scripts/create_kmhas_readable_obfuscation_dataset.py
```

### Baseline

- Input: clean Korean comments
- Character analyzer: `char_wb`
- N-gram range: 2-5
- Max features: 60,000
- Minimum document frequency: 2
- SVD dimensions: 160
- MLP hidden units: 96
- Batch size: 256
- Maximum iterations: 18
- Early stopping: enabled
- Random seed: 42

### Normalization-Aware Model

Training input:

```text
normalized clean train
+ normalized glyph-mix-35 train
```

Evaluation input:

```text
clean / romanized / roman+glyph
-> NFKC
-> glyph-to-Latin mapping
-> romanized Korean reconstruction
-> char TF-IDF MLP
```

Train or regenerate:

```bash
python scripts/train_evaluate_kmhas_tfidf_mlp.py \
  --models char_tfidf_mlp \
  --train-settings clean_only aug_glyph35 \
  --save-models

python scripts/train_evaluate_kmhas_defense_models.py --save-models
```

## 3. KoELECTRA-small

- Base model: `monologg/koelectra-small-v3-discriminator`
- Balanced clean K-MHaS train rows: 6,000
- Epochs: 1
- Batch size: 16
- Maximum length: 128
- Learning rate: `2e-5`
- Seed: 42

```bash
python scripts/train_evaluate_koelectra_free_llm_benchmark.py
```

Reuse the included checkpoint:

```bash
python scripts/train_evaluate_koelectra_free_llm_benchmark.py --reuse-saved
```

## 4. Qwen Raw and Calibrated Evaluation

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime: MLX-LM on Apple Silicon
- Output score: next-token logit difference between label `1` and label `0`
- Calibration: threshold selected on the separate validation benchmark
- Threshold objective: minimum balanced error, then higher Macro F1, then lower
  max failure rate

```bash
python scripts/evaluate_qwen_mlx_free_llm_benchmark.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input-mode raw \
  --run-name qwen_raw \
  --resume
```

## 5. Qwen LoRA Recipe

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Training examples: 12,000
- Validation examples: 1,500
- Training variants: clean, romanized 35, roman+glyph 35
- Held-out stress variants: romanized 70, roman+glyph 70
- LoRA rank: 8
- Scale: 2.0, equivalent alpha 16
- Adapted layers: 16
- Batch size: 8
- Gradient accumulation: 1
- Iterations: 4,500
- Epochs: 3
- Maximum sequence length: 256
- Learning rate: peak `2e-5`
- Warmup: 100 iterations from `1e-6`
- Schedule: cosine decay to `2e-6`
- AdamW epsilon: `1e-6`
- Weight decay: 0
- Gradient checkpointing: enabled
- Checkpoints: 1,500, 3,000, 4,500
- Selected checkpoint: 4,500
- Seed: 20260609

```bash
mlx_lm.lora --config configs/qwen_lora_mlx.yaml
python scripts/prepare_qwen_lora_checkpoint_dirs.py
python scripts/select_best_qwen_lora_checkpoint.py
```

The initial `1e-4` trial was excluded because long examples could remove the
assistant label during truncation, creating zero-supervision batches and NaN
loss. The final run uses token-aware head-and-tail truncation and verifies that
the supervised label tokens remain inside the 256-token sequence.

## 6. Primary Local Results

| Rank | System | Worst BA | Mean BA | Mean Macro F1 | Worst max(FNR, FPR) |
|---:|---|---:|---:|---:|---:|
| 1 | Normalization + Char TF-IDF MLP | 0.772 | 0.797 | 0.797 | 0.312 |
| 2 | Qwen LoRA + normalization + calibration | 0.672 | 0.716 | 0.715 | 0.396 |
| 3 | Char TF-IDF MLP | 0.572 | 0.699 | 0.666 | 0.848 |
| 4 | KoELECTRA-small | 0.544 | 0.659 | 0.632 | 0.880 |
| 5 | Qwen raw zero-shot | 0.522 | 0.562 | 0.510 | 0.852 |

### Qwen on Roman+Glyph 70

| Qwen stage | Balanced accuracy | Macro F1 | FNR | FPR | Predicted hate rate |
|---|---:|---:|---:|---:|---:|
| Raw zero-shot | 0.522 | 0.444 | 0.104 | 0.852 | 0.874 |
| Raw calibrated | 0.548 | 0.546 | 0.392 | 0.512 | 0.560 |
| Normalized calibrated | 0.532 | 0.486 | 0.768 | 0.168 | 0.200 |
| LoRA + normalized + calibrated | 0.672 | 0.671 | 0.268 | 0.388 | 0.560 |

Interpretation:

- raw Qwen catches most hate comments but over-blocks non-hate comments;
- normalization alone reduces over-blocking but increases missed hate;
- LoRA plus calibration produces the best Qwen balance;
- the normalization-aware character model remains the strongest local system
  under the defined worst-case ranking.

## 7. Supplementary Free UI Comparison

This experiment uses 50 shared base comments and five variants, for 250 rows per
service. It is separate from the 2,500-row primary benchmark.

| Rank | Free UI system | Worst BA | Mean BA | Worst max failure |
|---:|---|---:|---:|---:|
| 1 | Gemini Free UI | 0.860 | 0.892 | 0.200 |
| 2 | Claude Free UI | 0.800 | 0.832 | 0.360 |
| 3 | ChatGPT Free UI | 0.780 | 0.820 | 0.400 |
| 4 | Grok Free UI | 0.700 | 0.796 | 0.480 |

Important limitation: exact backend model names were not recorded, and free UI
services may change models or fallback behavior. These results are supplementary
and must not be merged into the main leaderboard.

## 8. Result Files

```text
outputs/kmhas_free_llm_benchmark/reports/main_system_ranking.csv
outputs/kmhas_free_llm_benchmark/reports/all_model_metrics.csv
outputs/kmhas_free_llm_benchmark/reports/qwen_improvement_key_numbers.csv
outputs/kmhas_free_llm_benchmark/reports/best_tested_system.csv
outputs/kmhas_free_llm_benchmark/reports/external_ui_250_ranking.csv
outputs/kmhas_free_llm_benchmark/reports/benchmark_integrity_report.json
outputs/kmhas_free_llm_benchmark/figures/
```
