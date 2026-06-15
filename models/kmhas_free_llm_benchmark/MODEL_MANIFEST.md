# Shared Model Artifacts

## KoELECTRA-small reference

- Directory: `koelectra_small/`
- Base model: `monologg/koelectra-small-v3-discriminator`
- Fine-tuning data: balanced K-MHaS clean train subset 6,000 rows
- Epoch: 1
- Learning rate: `2e-5`
- Max length: 128
- Output: binary hate / non-hate classifier

Load:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "models/kmhas_free_llm_benchmark/koelectra_small"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
```

## Qwen2.5-1.5B LoRA

- Final adapter directory: `qwen_lora/best/`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base weights are not duplicated in this project because they are much larger than the adapter.
- Training recipe: `../../configs/qwen_lora_mlx.yaml`
- Training examples: clean, romanized 35, roman+glyph 35
- Evaluation-only generalization variants: romanized 70, roman+glyph 70
- Completed training: 4,500 steps, 3 epochs
- Selected checkpoint: 4,500 steps
- Selection rule: highest calibration balanced accuracy, then lower max(FNR, FPR), then higher macro F1
- Calibration balanced accuracy: 0.7364
- Final `roman_glyph_70`: balanced accuracy 0.672, FNR 0.268, FPR 0.388

Load with MLX-LM:

```python
from mlx_lm import load

model, tokenizer = load(
    "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="models/kmhas_free_llm_benchmark/qwen_lora/best",
)
```

## Character TF-IDF MLP release assets

- `char_tfidf_mlp_clean_only_model.zip`
- `normalization_char_tfidf_mlp_model.zip`

These two approximately 76 MB joblib models are distributed separately as
GitHub Release assets. See `docs/MODELS.md` for extraction paths and recipes.

Failed and interrupted Qwen training runs are not distributed.
