# Korean Hate-Speech Robustness under Text Obfuscation

This repository evaluates how Korean hate-speech classifiers behave as comments
are increasingly obfuscated with romanized Korean and rune-like glyphs. It also
tests two complementary defenses:

1. rule-based normalization followed by a character TF-IDF MLP;
2. normalization, validation-set calibration, and LoRA adaptation of
   Qwen2.5-1.5B-Instruct.

The central safety question is not only whether a model detects hate speech, but
whether it avoids both:

- **under-detection**: missing actual hate comments;
- **over-blocking**: incorrectly blocking non-hate comments.

## Main Result

The primary benchmark uses the same 500 K-MHaS test comments, balanced as
250 hate and 250 non-hate, across five variants. Each local system therefore
produces 2,500 predictions.

| Rank | System | Worst-case balanced accuracy | Mean balanced accuracy | Worst max(FNR, FPR) |
|---:|---|---:|---:|---:|
| 1 | Normalization + Char TF-IDF MLP | 0.772 | 0.797 | 0.312 |
| 2 | Qwen LoRA + normalization + calibration | 0.672 | 0.716 | 0.396 |
| 3 | Char TF-IDF MLP | 0.572 | 0.699 | 0.848 |
| 4 | KoELECTRA-small | 0.544 | 0.659 | 0.880 |
| 5 | Qwen raw zero-shot | 0.522 | 0.562 | 0.852 |

On the strongest `roman_glyph_70` condition:

- raw Qwen: balanced accuracy `0.522`, FNR `0.104`, FPR `0.852`;
- improved Qwen: balanced accuracy `0.672`, FNR `0.268`, FPR `0.388`.

LoRA and calibration substantially reduced Qwen's over-blocking, while the
normalization-aware character model achieved the best worst-case balance on the
full local benchmark.

![Main ranking](outputs/kmhas_free_llm_benchmark/figures/final_system_ranking.png)

## Benchmark Variants

| Variant | Description |
|---|---|
| `clean` | Original Korean comment |
| `romanized_35` | About 35% of Hangul syllables replaced by romanized Korean |
| `romanized_70` | About 70% replaced by romanized Korean |
| `roman_glyph_35` | Mild romanization with partial rune/glyph substitution |
| `roman_glyph_70` | Strong romanization with heavier rune/glyph substitution |

The binary task uses the original K-MHaS labels:

```text
original label 8  -> non-hate
any other label  -> hate
```

## Repository Contents

```text
configs/                         MLX-LM LoRA configuration
docs/                            dataset, model, recipe, and upload documentation
models/                          Qwen LoRA adapter and fine-tuned KoELECTRA-small
outputs/kmhas_free_llm_benchmark/
  data/                          aligned benchmark and LoRA training data
  figures/                       final PNG figures
  predictions/                   per-system prediction CSV files
  reports/                       metrics, rankings, audits, and run metadata
scripts/                         preparation, training, evaluation, and summary code
```

## Used Models

- Character TF-IDF + MLP baseline
- Normalization-aware Character TF-IDF + MLP
- `monologg/koelectra-small-v3-discriminator`
- `Qwen/Qwen2.5-1.5B-Instruct`
- ChatGPT, Gemini, Claude, and Grok free web UIs as a supplementary 250-row
  comparison

The Qwen base weights are not redistributed. The repository includes only the
trained LoRA adapter. The two large classical `.joblib` files are distributed as
separate release assets; see [docs/MODELS.md](docs/MODELS.md).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_free_llm_benchmark.py
```

To reproduce the summary tables and figures from the included predictions:

```bash
MPLCONFIGDIR=/tmp/matplotlib-safe-ai \
XDG_CACHE_HOME=/tmp/xdg-cache-safe-ai \
python scripts/summarize_free_llm_benchmark.py

python scripts/summarize_external_ui_250_comparison.py
```

For complete training and evaluation commands, see
[docs/TRAINING_RECIPES_AND_RESULTS.md](docs/TRAINING_RECIPES_AND_RESULTS.md).

## Dataset Source and Caution

The source dataset is [K-MHaS](https://github.com/adlnlp/K-MHaS), a Korean
multi-label hate-speech dataset. The included benchmark contains selected
comments and deterministic derived variants.

The files contain real hate or offensive language. Keep the source citation and
review the upstream dataset terms before public redistribution. This project
does not claim ownership of the original comments.

## Interpretation Boundary

“Best” means the best tested system on this specific K-MHaS obfuscation
benchmark. It does not establish a universally best moderation model.

