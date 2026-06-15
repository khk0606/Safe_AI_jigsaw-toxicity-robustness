#!/usr/bin/env python3
"""Train K-MHaS TF-IDF+MLP baselines and evaluate glyph-mix robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_VARIANT_PATHS = {
    "clean": {
        "train": "data/kmhas/processed/clean/kmhas_train_clean.csv",
        "valid": "data/kmhas/processed/clean/kmhas_valid_clean.csv",
        "test": "data/kmhas/processed/clean/kmhas_test_clean.csv",
    },
    "glyph_mix_35": {
        "train": "data/kmhas/processed/glyph_mix_35/kmhas_train_glyph_mix_35.csv",
        "valid": "data/kmhas/processed/glyph_mix_35/kmhas_valid_glyph_mix_35.csv",
        "test": "data/kmhas/processed/glyph_mix_35/kmhas_test_glyph_mix_35.csv",
    },
    "glyph_mix_90": {
        "train": "data/kmhas/processed/glyph_mix_90/kmhas_train_glyph_mix_90.csv",
        "valid": "data/kmhas/processed/glyph_mix_90/kmhas_valid_glyph_mix_90.csv",
        "test": "data/kmhas/processed/glyph_mix_90/kmhas_test_glyph_mix_90.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kmhas_glyph_mix"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/kmhas_glyph_mix"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["word_tfidf_mlp", "char_tfidf_mlp"],
        choices=["word_tfidf_mlp", "char_tfidf_mlp"],
    )
    parser.add_argument(
        "--train-settings",
        nargs="+",
        default=["clean_only", "aug_glyph35"],
        choices=["clean_only", "aug_glyph35", "aug_glyph90", "aug_glyph35_glyph90"],
    )
    parser.add_argument(
        "--eval-variants",
        nargs="+",
        default=["clean", "glyph_mix_35", "glyph_mix_90"],
        choices=["clean", "glyph_mix_35", "glyph_mix_90"],
    )
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--svd-components", type=int, default=160)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Debug only. 0 means use all rows.",
    )
    return parser.parse_args()


def read_split(variant: str, split: str) -> pd.DataFrame:
    path = Path(DEFAULT_VARIANT_PATHS[variant][split])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    expected = {"document", "obfuscated_document", "is_hate_speech", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def normalized_frame(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": df[text_column].fillna("").astype(str),
            "label": df["is_hate_speech"].astype(int),
            "source_id": df.get("id", pd.Series(range(len(df)))).astype(str),
            "original_label": df["label"].astype(str),
        }
    )


def build_train_frame(train_setting: str, max_train_rows: int, seed: int) -> pd.DataFrame:
    clean_train = normalized_frame(read_split("clean", "train"), "document")
    frames = [clean_train.assign(train_source="clean")]

    if train_setting in {"aug_glyph35", "aug_glyph35_glyph90"}:
        glyph35_train = normalized_frame(read_split("glyph_mix_35", "train"), "obfuscated_document")
        frames.append(glyph35_train.assign(train_source="glyph_mix_35"))

    if train_setting in {"aug_glyph90", "aug_glyph35_glyph90"}:
        glyph90_train = normalized_frame(read_split("glyph_mix_90", "train"), "obfuscated_document")
        frames.append(glyph90_train.assign(train_source="glyph_mix_90"))

    train_df = pd.concat(frames, ignore_index=True)
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = (
            train_df.groupby("label", group_keys=False)
            .sample(n=max_train_rows // 2, random_state=seed)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    else:
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df


def make_pipeline(model_name: str, args: argparse.Namespace) -> Pipeline:
    if model_name == "word_tfidf_mlp":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=args.max_features,
            min_df=2,
            max_df=0.98,
            lowercase=True,
            strip_accents=None,
            sublinear_tf=True,
        )
    elif model_name == "char_tfidf_mlp":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=args.max_features,
            min_df=2,
            max_df=0.98,
            lowercase=True,
            strip_accents=None,
            sublinear_tf=True,
        )
    else:
        raise ValueError(model_name)

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("svd", TruncatedSVD(n_components=args.svd_components, random_state=args.seed)),
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(args.hidden_size,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=args.batch_size,
                    learning_rate_init=1e-3,
                    max_iter=args.epochs,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=3,
                    random_state=args.seed,
                    verbose=False,
                ),
            ),
        ]
    )


def predict_proba_positive(model: Pipeline, texts: pd.Series) -> np.ndarray:
    proba = model.predict_proba(texts)
    return np.asarray(proba[:, 1])


def metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["auroc"] = roc_auc_score(y_true, proba)
    except ValueError:
        out["auroc"] = float("nan")
    return out


def evaluate_model(
    model: Pipeline,
    model_name: str,
    train_setting: str,
    eval_variants: list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    eval_frames = {
        variant: normalized_frame(read_split(variant, "test"), "obfuscated_document")
        for variant in eval_variants
    }
    if "clean" not in eval_frames:
        raise ValueError("clean must be included in eval variants for drop metrics.")

    clean_df = eval_frames["clean"]
    y_clean = clean_df["label"].to_numpy(dtype=int)
    clean_proba = predict_proba_positive(model, clean_df["text"])
    clean_pred = (clean_proba >= 0.5).astype(int)
    clean_correct = clean_pred == y_clean
    clean_hate_mask = y_clean == 1
    clean_metrics = metrics(y_clean, clean_proba)

    rows = []
    failure_rows = []
    for variant, eval_df in eval_frames.items():
        y = eval_df["label"].to_numpy(dtype=int)
        if not np.array_equal(y, y_clean):
            raise ValueError(f"Label mismatch between clean and {variant}.")
        proba = clean_proba if variant == "clean" else predict_proba_positive(model, eval_df["text"])
        pred = (proba >= 0.5).astype(int)
        m = metrics(y, proba)

        clean_correct_count = int(clean_correct.sum())
        attack_success_count = int((clean_correct & (pred != y)).sum())
        hate_count = int(clean_hate_mask.sum())
        hate_conf_drop = float(np.mean(clean_proba[clean_hate_mask] - proba[clean_hate_mask])) if hate_count else 0.0

        rows.append(
            {
                "model": model_name,
                "train_setting": train_setting,
                "eval_variant": variant,
                "train_rows": None,
                "eval_rows": len(eval_df),
                "hate_rows": int(y.sum()),
                "not_hate_rows": int(len(y) - y.sum()),
                "f1_drop_from_clean": clean_metrics["f1"] - m["f1"],
                "recall_drop_from_clean": clean_metrics["recall"] - m["recall"],
                "attack_success_rate": attack_success_count / clean_correct_count if clean_correct_count else 0.0,
                "confidence_shift": float(np.mean(np.abs(clean_proba - proba))),
                "hate_confidence_drop": hate_conf_drop,
                **m,
            }
        )

        if variant != "clean":
            failed_idx = np.where(clean_correct & (pred != y))[0][:50]
            for idx in failed_idx:
                failure_rows.append(
                    {
                        "model": model_name,
                        "train_setting": train_setting,
                        "eval_variant": variant,
                        "row_id": clean_df["source_id"].iloc[idx],
                        "label": int(y[idx]),
                        "clean_probability": float(clean_proba[idx]),
                        "variant_probability": float(proba[idx]),
                        "clean_prediction": int(clean_pred[idx]),
                        "variant_prediction": int(pred[idx]),
                        "probability_delta": float(clean_proba[idx] - proba[idx]),
                    }
                )

    return rows, pd.DataFrame(failure_rows)


def write_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# K-MHaS Glyph-Mix TF-IDF+MLP Results",
        "",
        "## Main Metrics",
        "",
        "| Model | Train | Eval | Accuracy | Precision | Recall | F1 | AUROC | F1 Drop | Recall Drop | ASR | Hate Conf Drop |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = metrics_df.sort_values(["model", "train_setting", "eval_variant"])
    for _, row in ordered.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["train_setting"]),
                    str(row["eval_variant"]),
                    fmt(row["accuracy"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["f1"]),
                    fmt(row["auroc"]),
                    fmt(row["f1_drop_from_clean"]),
                    fmt(row["recall_drop_from_clean"]),
                    fmt(row["attack_success_rate"]),
                    fmt(row["hate_confidence_drop"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `F1 Drop`: clean test F1 minus glyph-mix test F1 under the same trained model.",
            "- `Recall Drop`: clean hate recall minus glyph-mix hate recall.",
            "- `ASR`: among samples correctly predicted on clean test, the fraction that becomes wrong under glyph-mix.",
            "- `Hate Conf Drop`: average decrease in hate-class probability for true hate samples.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_failures: list[pd.DataFrame] = []
    run_models: list[dict[str, object]] = []

    for model_name in args.models:
        for train_setting in args.train_settings:
            train_df = build_train_frame(train_setting, args.max_train_rows, args.seed)
            print(
                f"Training model={model_name} train_setting={train_setting} "
                f"rows={len(train_df)} positives={int(train_df['label'].sum())}"
            )
            model = make_pipeline(model_name, args)
            model.fit(train_df["text"], train_df["label"])
            rows, failures = evaluate_model(model, model_name, train_setting, args.eval_variants)
            for row in rows:
                row["train_rows"] = len(train_df)
            all_rows.extend(rows)
            if not failures.empty:
                all_failures.append(failures)

            mlp = model.named_steps["mlp"]
            run_models.append(
                {
                    "model": model_name,
                    "train_setting": train_setting,
                    "train_rows": len(train_df),
                    "mlp_n_iter": int(getattr(mlp, "n_iter_", -1)),
                    "mlp_loss": float(getattr(mlp, "loss_", np.nan)),
                    "tfidf_vocab_size": len(getattr(model.named_steps["tfidf"], "vocabulary_", {})),
                }
            )
            if args.save_models:
                model_path = args.models_dir / f"{model_name}_{train_setting}.joblib"
                joblib.dump(model, model_path)
                print(f"Saved {model_path}")

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = reports_dir / "tfidf_mlp_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    failure_path = reports_dir / "tfidf_mlp_failure_cases_masked.csv"
    if all_failures:
        pd.concat(all_failures, ignore_index=True).to_csv(failure_path, index=False)
    else:
        pd.DataFrame().to_csv(failure_path, index=False)

    run_info = {
        "args": vars(args),
        "models": run_models,
        "note": "Failure case file excludes raw text to avoid exposing hate/offensive rows.",
    }
    (reports_dir / "tfidf_mlp_run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_summary(metrics_df, reports_dir / "tfidf_mlp_summary.md")

    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved summary to {reports_dir / 'tfidf_mlp_summary.md'}")


if __name__ == "__main__":
    main()

