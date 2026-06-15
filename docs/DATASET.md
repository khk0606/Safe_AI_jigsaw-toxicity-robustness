# Dataset

## 1. Source

- Name: **K-MHaS**
- Repository: https://github.com/adlnlp/K-MHaS
- Paper: *K-MHaS: A Multi-label Hate Speech Detection Dataset in Korean
  Online News Comment*
- Original task: Korean multi-label hate-speech classification

The original repository provides train, validation, and test TSV files. They can
be downloaded from the upstream source and placed under:

```text
data/kmhas/raw/
  kmhas_train.txt
  kmhas_valid.txt
  kmhas_test.txt
```

## 2. Binary Label Conversion

This project converts the original multi-label task into binary classification.

```text
original label includes 8 -> non-hate (0)
otherwise                 -> hate (1)
```

The conversion is implemented in:

```text
scripts/prepare_kmhas_clean_dataset.py
```

## 3. Obfuscation Generation

The clean Korean comments are deterministically transformed into four variants:

| Variant | Hangul replacement target | Glyph mixing |
|---|---:|---:|
| `romanized_35` | 35% | 0% |
| `romanized_70` | 70% | 0% |
| `roman_glyph_35` | 35% | 35% of generated Latin letters |
| `roman_glyph_70` | 70% | 45% of generated Latin letters |

Example transformation:

```text
부산의 자랑
-> busanui jarang
-> ᛒᚢsanui jᚨrang
```

Generation code:

```text
scripts/create_kmhas_readable_obfuscation_dataset.py
```

The transformation uses a fixed seed and a per-row hash, so the same source row
produces the same result.

## 4. Common Evaluation Benchmark

The included test benchmark contains:

- 500 unique test comments
- 250 hate and 250 non-hate
- 5 variants per comment
- 2,500 total rows per system

The calibration benchmark has the same size and class balance but uses the
validation split. It is used only to choose Qwen's decision threshold.

Files:

```text
outputs/kmhas_free_llm_benchmark/data/benchmark_test_2500.csv
outputs/kmhas_free_llm_benchmark/data/benchmark_calibration_2500.csv
outputs/kmhas_free_llm_benchmark/data/benchmark_test_ids.csv
outputs/kmhas_free_llm_benchmark/data/benchmark_calibration_ids.csv
outputs/kmhas_free_llm_benchmark/data/benchmark_metadata.json
```

## 5. Qwen LoRA Dataset

- 4,000 unique K-MHaS training IDs
- Training variants: clean, romanized 35, roman+glyph 35
- 12,000 training examples
- 500 unique validation IDs and 1,500 validation examples
- 50:50 hate/non-hate balance
- `romanized_70` and `roman_glyph_70` are held out from training

Files:

```text
outputs/kmhas_free_llm_benchmark/data/qwen_lora/train.jsonl
outputs/kmhas_free_llm_benchmark/data/qwen_lora/valid.jsonl
outputs/kmhas_free_llm_benchmark/data/qwen_lora/train_examples.csv
outputs/kmhas_free_llm_benchmark/data/qwen_lora/valid_examples.csv
```

## 6. Leakage and Integrity Checks

The following ID intersections were verified as zero:

- test vs calibration
- test vs LoRA train
- test vs LoRA validation
- calibration vs LoRA train
- calibration vs LoRA validation
- LoRA train vs LoRA validation

See:

```text
outputs/kmhas_free_llm_benchmark/reports/benchmark_integrity_report.json
```

## 7. Redistribution Note

The included CSV files contain selected K-MHaS comments and deterministic
derived text. They may contain offensive language. Keep the K-MHaS citation
with redistributed copies and review the current upstream terms before making
the files public.

