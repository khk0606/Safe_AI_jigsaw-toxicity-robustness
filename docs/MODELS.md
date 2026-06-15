# Used Models and Artifacts

## 1. Character TF-IDF + MLP

Two scikit-learn pipelines were used:

- `char_tfidf_mlp_clean_only.joblib`
- `normalization_char_tfidf_mlp.joblib`

Pipeline:

```text
text
-> character n-gram TF-IDF, 2-5 grams
-> TruncatedSVD, 160 dimensions
-> StandardScaler
-> one-hidden-layer MLP, 96 units
-> hate probability
```

The normalization model receives text after:

```text
Unicode NFKC
-> rune/glyph to Latin mapping
-> romanized Korean parsing
-> Hangul-like reconstruction
-> character classifier
```

Because each `.joblib` file is about 76 MB, they are packaged as separate
GitHub Release assets:

```text
char_tfidf_mlp_clean_only_model.zip
normalization_char_tfidf_mlp_model.zip
```

After download, extract them to:

```text
models/kmhas_glyph_mix/char_tfidf_mlp_clean_only.joblib
models/kmhas_glyph_mix/defense_models/normalization_char_tfidf_mlp.joblib
```

They can also be regenerated with the included training scripts.

## 2. KoELECTRA-small

- Base: https://huggingface.co/monologg/koelectra-small-v3-discriminator
- Task: binary hate/non-hate classification
- Fine-tuning data: balanced clean K-MHaS subset, 6,000 rows
- Epoch: 1
- Learning rate: `2e-5`
- Max length: 128

Included path:

```text
models/kmhas_free_llm_benchmark/koelectra_small/
```

Load:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "models/kmhas_free_llm_benchmark/koelectra_small"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
```

## 3. Qwen2.5-1.5B LoRA

- Base: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Included artifact: locally trained LoRA adapter only
- Best checkpoint: step 4,500

Included path:

```text
models/kmhas_free_llm_benchmark/qwen_lora/best/
```

Load with MLX-LM:

```python
from mlx_lm import load

model, tokenizer = load(
    "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="models/kmhas_free_llm_benchmark/qwen_lora/best",
)
```

The Qwen base weights are not redistributed.

## 4. External Free UI Systems

The supplementary experiment evaluated:

- ChatGPT Free UI
- Gemini Free UI
- Claude Free UI
- Grok Free UI

These are prediction records, not downloadable model artifacts. The exact
backend model names were not recorded by the teammate, so the results are
reported only as free UI service results run on June 13, 2026.

## 5. Artifact Integrity

SHA-256 hashes are provided in:

```text
CHECKSUMS_SHA256.txt
```

