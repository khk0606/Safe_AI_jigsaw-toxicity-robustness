# Robustness Evaluation of Toxicity Detection under Text Perturbations

This project evaluates whether toxicity classifiers remain reliable when toxic comments are obfuscated with small text perturbations.

The current project upload focuses on:

```text
TextCNN clean baseline
TextCNN perturbation-augmented defense models
RoBERTa SOTA-style reference model
```

The presentation deck is submitted separately. This repository contains the shareable dataset, model artifacts, training recipes, result tables, and figures.

## Project Motivation

Online moderation systems should detect toxic comments even when users slightly modify words to bypass filters.

Examples:

```text
idiot  -> i d i o t
hate   -> h@te
bad    -> bad!!!
```

The key question is:

```text
Does a model that performs well on clean comments still work after realistic text obfuscation?
```

## Repository Files

| File | Description |
|---|---|
| `jigsaw_toxicity_robustness_shareable.zip` | Shareable Jigsaw subset, targeted augmented training set, and clean/perturbed evaluation set |
| `models_textcnn_core.zip` | TextCNN checkpoints and RoBERTa reference information |
| `training_recipes_results.zip` | Training scripts, evaluation scripts, result tables, figures, and explanations |

## Dataset

Source dataset:

```text
Jigsaw Unintended Bias in Toxicity Classification
```

Mirror used during this project:

```text
Intuit-GenSRF/jigsaw-unintended-bias
```

Binary label rule:

```text
target >= 0.5 -> toxic
target < 0.5  -> non-toxic
```

Balanced split:

| Split | Non-toxic | Toxic | Total |
|---|---:|---:|---:|
| Train | 5,000 | 5,000 | 10,000 |
| Validation | 1,000 | 1,000 | 2,000 |
| Test | 1,000 | 1,000 | 2,000 |

The dataset zip includes:

```text
jigsaw_subset_14k_clean.csv
train_augmented_targeted_100.csv
test_clean_and_perturbed_eval.csv
manifest.json
column_schema.csv
reproducibility/perturbations.py
```

## Perturbations

The perturbation generator changes the text surface form while keeping the original label fixed.

| Perturbation | Example |
|---|---|
| typo | `stupid -> stupdi` |
| character substitution | `hate -> h@te` |
| spacing | `idiot -> i d i o t` |
| repetition | `bad -> baaad` |
| punctuation noise | `bad -> bad!!!` |
| combined | multiple perturbations mixed |

## Models

### TextCNN

The project trains and evaluates TextCNN variants:

| Model | Role |
|---|---|
| TextCNN clean | Baseline trained on clean comments |
| TextCNN 25% general augmentation | Augmentation-ratio comparison |
| TextCNN 100% targeted augmentation | Main robustness defense model |

TextCNN architecture:

```text
Embedding
-> Conv1D kernel sizes [3, 4, 5]
-> Global max pooling
-> Dropout
-> Linear output layer
```

### RoBERTa Reference

Public SOTA-style reference:

```text
unitary/unbiased-toxic-roberta
https://huggingface.co/unitary/unbiased-toxic-roberta
```

The RoBERTa checkpoint is not uploaded because it is publicly available. It is used as a strong clean-performance reference model.

## Metrics

| Metric | Meaning | Better Direction |
|---|---|---|
| Clean F1 | F1 score on original clean text | Higher |
| Perturbed F1 | F1 score after text perturbation | Higher |
| F1 Drop | Clean F1 - perturbed F1 | Lower |
| ASR | Clean-correct samples that become wrong after perturbation | Lower |
| Toxic Recall Drop | Drop in recall on toxic-only samples | Lower |

## Main Results

### Clean Performance

| Model | Clean F1 | AUROC |
|---|---:|---:|
| RoBERTa reference | 0.830 | 0.987 |
| TextCNN clean | 0.814 | 0.901 |
| TextCNN targeted augmentation | 0.827 | 0.904 |

### Combined Perturbation Robustness

| Model | F1 Drop | ASR |
|---|---:|---:|
| RoBERTa reference | 0.117 | 10.8% |
| TextCNN clean | 0.086 | 13.2% |
| TextCNN targeted augmentation | 0.028 | 5.7% |

### Toxic-only Evaluation

| Model | Clean Toxic Recall | Perturbed Toxic Recall | Recall Drop | ASR / Evasion |
|---|---:|---:|---:|---:|
| RoBERTa reference | 0.718 | 0.566 | 0.152 | 10.8% |
| TextCNN clean | 0.828 | 0.676 | 0.152 | 13.2% |
| TextCNN targeted augmentation | 0.846 | 0.804 | 0.042 | 5.7% |

## Key Takeaway

RoBERTa has the strongest clean performance, but targeted TextCNN is more stable under this controlled text-obfuscation perturbation setting.

This does not mean TextCNN is generally better than RoBERTa. The claim is narrower:

```text
targeted perturbation augmentation improves robustness under the tested moderation-evasion threat model.
```

## Reproducibility

The full training and evaluation details are in:

```text
training_recipes_results.zip
```

It includes:

```text
scripts/
reports/
figures/
requirements.txt
README.md
```

The model checkpoints are in:

```text
models_textcnn_core.zip
```
