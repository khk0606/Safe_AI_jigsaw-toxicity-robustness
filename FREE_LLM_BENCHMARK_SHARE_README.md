# Free LLM Robustness Benchmark Share Package

## Included

- aligned 500-ID / 2,500-row test and calibration benchmark
- LoRA train/validation JSONL and manifests
- model prediction CSV files
- integrated metrics, checkpoint selection, and integrity audit
- final PNG figures
- Qwen2.5-1.5B best LoRA adapter
- fine-tuned KoELECTRA-small model
- ChatGPT, Gemini, Claude, and Grok free-UI supplementary predictions
- experiment scripts and MLX-LM config
- Korean result and reproduction documents

## Not Included

- Qwen2.5-1.5B base weights: download `Qwen/Qwen2.5-1.5B-Instruct`
- large classical `.joblib` files in the repository ZIP. They are supplied as
  separate model release assets:
  - `char_tfidf_mlp_clean_only_model.zip`
  - `normalization_char_tfidf_mlp_model.zip`
- invalid or interrupted Qwen training runs

The classical model files are approximately 76 MB each and remain available at the paths above in the full local project.

## Main Result

- Best tested system: normalization + Char TF-IDF MLP
- Worst-case balanced accuracy: 0.772
- Final Qwen worst-case balanced accuracy: 0.672
- Raw Qwen worst-case balanced accuracy: 0.522

## Start Here

1. Read `README.md` or `README_KO.md`.
2. Review `docs/TRAINING_RECIPES_AND_RESULTS.md`.
3. Check `outputs/kmhas_free_llm_benchmark/reports/benchmark_integrity_report.json`.
4. Use `docs/MODELS.md` to load the shared models.

The dataset contains hate/offensive text. Keep `KMHAS_BENCHMARK_SHARING_NOTE.md` with all redistributed copies.
