# K-MHaS Benchmark Sharing Note

## Source

- Dataset: K-MHaS
- Repository: https://github.com/adlnlp/K-MHaS
- Paper: *K-MHaS: A Multi-label Hate Speech Detection Dataset in Korean Online News Comment*

## Contents

The benchmark CSV files contain selected K-MHaS comments and deterministic derived variants:

- clean
- romanized 35 / 70
- roman + glyph 35 / 70

Labels are converted to a binary task:

- original label `8`: non-hate
- any other original label: hate

## Sharing Caution

The files contain real hate/offensive language and derived text based on the upstream dataset. Keep the source citation with every copy. Before public redistribution, review the current K-MHaS repository terms and institutional requirements. This project does not claim ownership of the original comments.

Qwen base weights are not included. The shared model artifact contains only the locally trained LoRA adapter and requires `Qwen/Qwen2.5-1.5B-Instruct` to run.
