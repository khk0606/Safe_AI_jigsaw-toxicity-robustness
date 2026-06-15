#!/usr/bin/env python3
"""Train/evaluate K-MHaS defense models for glyph-mix robustness.

Defense A: normalization-aware char TF-IDF MLP.
Defense B: obfuscation-score router between clean and robust classifiers.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from functools import lru_cache
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

CHOSEONG = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
JUNGSEONG = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]
JONGSEONG = [
    "",
    "g",
    "kk",
    "gs",
    "n",
    "nj",
    "nh",
    "d",
    "l",
    "lg",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "b",
    "bs",
    "s",
    "ss",
    "ng",
    "j",
    "ch",
    "k",
    "t",
    "p",
    "h",
]

LATIN_TO_GLYPH = {
    "a": "ᚨ",
    "b": "ᛒ",
    "c": "ᚲ",
    "d": "ᛞ",
    "e": "ᛖ",
    "f": "ᚠ",
    "g": "ᚷ",
    "h": "ᚺ",
    "i": "ᛁ",
    "j": "ᛃ",
    "k": "ᚴ",
    "l": "ᛚ",
    "m": "ᛗ",
    "n": "ᚾ",
    "o": "ᛟ",
    "p": "ᛈ",
    "q": "ᛩ",
    "r": "ᚱ",
    "s": "ᛋ",
    "t": "ᛏ",
    "u": "ᚢ",
    "v": "ᚡ",
    "w": "ᚹ",
    "x": "ᛪ",
    "y": "ᚣ",
    "z": "ᛉ",
}
GLYPH_TO_LATIN = {value: key for key, value in LATIN_TO_GLYPH.items()}

LATIN_RUN_RE = re.compile(r"[A-Za-z0-9ŋ]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kmhas_glyph_mix/defense_models"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/kmhas_glyph_mix/defense_models"))
    parser.add_argument("--baseline-model-dir", type=Path, default=Path("models/kmhas_glyph_mix"))
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--svd-components", type=int, default=160)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0)
    return parser.parse_args()


@lru_cache(maxsize=1)
def roman_to_hangul_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    base = 0xAC00
    for cho_idx, cho in enumerate(CHOSEONG):
        for jung_idx, jung in enumerate(JUNGSEONG):
            for jong_idx, jong in enumerate(JONGSEONG):
                roman = cho + jung + jong
                if not roman:
                    continue
                syllable = chr(base + (cho_idx * 21 + jung_idx) * 28 + jong_idx)
                mapping.setdefault(roman, syllable)
    return mapping


@lru_cache(maxsize=1)
def roman_keys_by_len() -> list[str]:
    return sorted(roman_to_hangul_map(), key=len, reverse=True)


def glyphs_to_latin(text: str) -> str:
    out = []
    for char in text:
        out.append(GLYPH_TO_LATIN.get(char, char))
    return "".join(out).replace("ŋ", "ng")


def roman_run_to_hangul(run: str) -> str:
    run = run.lower()
    # Preserve short English/number fragments such as "tv" or "100".
    if len(run) < 2 or run.isdigit():
        return run

    mapping = roman_to_hangul_map()
    keys = roman_keys_by_len()
    i = 0
    out = []
    converted_chars = 0
    while i < len(run):
        matched = False
        for key in keys:
            if run.startswith(key, i):
                out.append(mapping[key])
                i += len(key)
                converted_chars += len(key)
                matched = True
                break
        if not matched:
            out.append(run[i])
            i += 1
    converted = "".join(out)
    # Avoid aggressively converting ordinary long English words when little was parsed.
    if converted_chars / max(1, len(run)) < 0.55:
        return run
    return converted


def normalize_obfuscated_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = glyphs_to_latin(text)
    # Frequent visual substitutions from the generated perturbation format.
    text = text.replace("0l", "이").replace("0L", "이")
    text = text.replace("ㄱr", "가").replace("ㄴr", "나").replace("ㄷr", "다")
    text = text.replace("ㅁr", "마").replace("ㅂr", "바").replace("ㅅr", "사")
    text = text.replace("ㅇr", "아").replace("zㅏ", "자").replace("hㅏ", "하")
    return LATIN_RUN_RE.sub(lambda m: roman_run_to_hangul(m.group(0)), text)


def obfuscation_score(text: str) -> float:
    text = str(text)
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    total = len(visible)
    rune = sum(0x16A0 <= ord(char) <= 0x16FF for char in visible) / total
    latin = sum(char.isascii() and char.isalpha() for char in visible) / total
    digit = sum(char.isdigit() for char in visible) / total
    jamo = sum(0x3130 <= ord(char) <= 0x318F for char in visible) / total
    punct = sum(not char.isalnum() and not (0xAC00 <= ord(char) <= 0xD7A3) for char in visible) / total
    return min(1.0, rune * 1.4 + latin * 0.25 + digit * 0.2 + jamo * 0.35 + punct * 0.1)


def read_split(variant: str, split: str) -> pd.DataFrame:
    path = Path(DEFAULT_VARIANT_PATHS[variant][split])
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def frame_for_variant(variant: str, split: str, text_mode: str) -> pd.DataFrame:
    raw = read_split(variant, split)
    text_col = "document" if variant == "clean" else "obfuscated_document"
    texts = raw[text_col].fillna("").astype(str)
    if text_mode == "normalized":
        texts = texts.map(normalize_obfuscated_text)
    elif text_mode != "raw":
        raise ValueError(text_mode)
    return pd.DataFrame(
        {
            "text": texts,
            "raw_text": raw[text_col].fillna("").astype(str),
            "label": raw["is_hate_speech"].astype(int),
            "source_id": raw.get("id", pd.Series(range(len(raw)))).astype(str),
        }
    )


def make_char_mlp(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    max_features=args.max_features,
                    min_df=2,
                    max_df=0.98,
                    lowercase=True,
                    strip_accents=None,
                    sublinear_tf=True,
                ),
            ),
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
                ),
            ),
        ]
    )


def predict_proba_positive(model: Pipeline, texts: pd.Series) -> np.ndarray:
    return np.asarray(model.predict_proba(texts)[:, 1])


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


def build_normalized_train_frame(args: argparse.Namespace) -> pd.DataFrame:
    clean = frame_for_variant("clean", "train", "normalized").assign(train_source="clean")
    glyph35 = frame_for_variant("glyph_mix_35", "train", "normalized").assign(train_source="glyph_mix_35_normalized")
    train = pd.concat([clean, glyph35], ignore_index=True)
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = (
            train.groupby("label", group_keys=False)
            .sample(n=args.max_train_rows // 2, random_state=args.seed)
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )
    else:
        train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    return train


def evaluate_proba(
    model_name: str,
    train_setting: str,
    eval_variant: str,
    train_rows: int,
    y_clean: np.ndarray,
    clean_proba: np.ndarray,
    proba: np.ndarray,
    extra: dict[str, float] | None = None,
) -> dict[str, object]:
    clean_metrics = metrics(y_clean, clean_proba)
    m = metrics(y_clean, proba)
    clean_pred = (clean_proba >= 0.5).astype(int)
    pred = (proba >= 0.5).astype(int)
    clean_correct = clean_pred == y_clean
    clean_hate_mask = y_clean == 1
    clean_correct_count = int(clean_correct.sum())
    row = {
        "model": model_name,
        "train_setting": train_setting,
        "eval_variant": eval_variant,
        "train_rows": train_rows,
        "eval_rows": len(y_clean),
        "hate_rows": int(y_clean.sum()),
        "not_hate_rows": int(len(y_clean) - y_clean.sum()),
        "f1_drop_from_clean": clean_metrics["f1"] - m["f1"],
        "recall_drop_from_clean": clean_metrics["recall"] - m["recall"],
        "attack_success_rate": int((clean_correct & (pred != y_clean)).sum()) / clean_correct_count
        if clean_correct_count
        else 0.0,
        "confidence_shift": float(np.mean(np.abs(clean_proba - proba))),
        "hate_confidence_drop": float(np.mean(clean_proba[clean_hate_mask] - proba[clean_hate_mask])),
        **m,
    }
    if extra:
        row.update(extra)
    return row


def train_normalization_defense(args: argparse.Namespace) -> tuple[Pipeline, pd.DataFrame]:
    train = build_normalized_train_frame(args)
    model = make_char_mlp(args)
    print(f"Training normalization_char_tfidf_mlp rows={len(train)} positives={int(train['label'].sum())}")
    model.fit(train["text"], train["label"])

    clean_test = frame_for_variant("clean", "test", "normalized")
    y_clean = clean_test["label"].to_numpy(dtype=int)
    clean_proba = predict_proba_positive(model, clean_test["text"])
    rows = []
    for variant in ["clean", "glyph_mix_35", "glyph_mix_90"]:
        eval_df = frame_for_variant(variant, "test", "normalized")
        proba = clean_proba if variant == "clean" else predict_proba_positive(model, eval_df["text"])
        rows.append(
            evaluate_proba(
                "normalization_char_tfidf_mlp",
                "clean_plus_glyph35_normalized",
                variant,
                len(train),
                y_clean,
                clean_proba,
                proba,
            )
        )
    return model, pd.DataFrame(rows)


def tune_router_threshold(
    clean_model: Pipeline,
    robust_model: Pipeline,
    thresholds: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    frames = []
    for variant in ["clean", "glyph_mix_35", "glyph_mix_90"]:
        valid_df = frame_for_variant(variant, "valid", "raw")
        frames.append(valid_df.assign(eval_variant=variant))
    valid = pd.concat(frames, ignore_index=True)
    y = valid["label"].to_numpy(dtype=int)
    clean_proba = predict_proba_positive(clean_model, valid["text"])
    robust_proba = predict_proba_positive(robust_model, valid["text"])
    scores = valid["raw_text"].map(obfuscation_score).to_numpy()

    rows = []
    for threshold in thresholds:
        route_robust = scores >= threshold
        proba = np.where(route_robust, robust_proba, clean_proba)
        m = metrics(y, proba)
        review_rate = float(route_robust.mean())
        utility = m["f1"] - 0.01 * review_rate
        rows.append(
            {
                "threshold": float(threshold),
                "utility": utility,
                "route_robust_rate": review_rate,
                "clean_route_rate": float((~route_robust).mean()),
                **m,
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(["utility", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def evaluate_router(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_path = args.baseline_model_dir / "char_tfidf_mlp_clean_only.joblib"
    robust_path = args.baseline_model_dir / "char_tfidf_mlp_aug_glyph35.joblib"
    if not clean_path.exists() or not robust_path.exists():
        raise FileNotFoundError("Baseline char_tfidf_mlp models must exist before router evaluation.")
    clean_model = joblib.load(clean_path)
    robust_model = joblib.load(robust_path)

    thresholds = np.linspace(0.02, 0.95, 48)
    best_threshold, threshold_table = tune_router_threshold(clean_model, robust_model, thresholds)
    print(f"Best router threshold={best_threshold:.4f}")

    clean_test = frame_for_variant("clean", "test", "raw")
    y_clean = clean_test["label"].to_numpy(dtype=int)
    clean_clean_proba = predict_proba_positive(clean_model, clean_test["text"])
    clean_robust_proba = predict_proba_positive(robust_model, clean_test["text"])
    clean_scores = clean_test["raw_text"].map(obfuscation_score).to_numpy()
    clean_route = clean_scores >= best_threshold
    clean_router_proba = np.where(clean_route, clean_robust_proba, clean_clean_proba)

    rows = []
    for variant in ["clean", "glyph_mix_35", "glyph_mix_90"]:
        eval_df = frame_for_variant(variant, "test", "raw")
        clean_model_proba = predict_proba_positive(clean_model, eval_df["text"])
        robust_model_proba = predict_proba_positive(robust_model, eval_df["text"])
        scores = eval_df["raw_text"].map(obfuscation_score).to_numpy()
        route_robust = scores >= best_threshold
        proba = np.where(route_robust, robust_model_proba, clean_model_proba)
        rows.append(
            evaluate_proba(
                "obfuscation_router_char_tfidf_mlp",
                "threshold_tuned_clean_glyph35_glyph90_valid",
                variant,
                78977 + 157954,
                y_clean,
                clean_router_proba,
                proba,
                extra={
                    "router_threshold": best_threshold,
                    "route_robust_rate": float(route_robust.mean()),
                    "mean_obfuscation_score": float(np.mean(scores)),
                },
            )
        )
    return pd.DataFrame(rows), threshold_table


def write_summary(metrics_df: pd.DataFrame, threshold_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# K-MHaS Defense Model Results",
        "",
        "## Metrics",
        "",
        "| Model | Train/Policy | Eval | Accuracy | Precision | Recall | F1 | AUROC | F1 Drop | Recall Drop | ASR | Hate Conf Drop | Route Robust Rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.sort_values(["model", "eval_variant"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["train_setting"]),
                    str(row["eval_variant"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['auroc']:.4f}",
                    f"{row['f1_drop_from_clean']:.4f}",
                    f"{row['recall_drop_from_clean']:.4f}",
                    f"{row['attack_success_rate']:.4f}",
                    f"{row['hate_confidence_drop']:.4f}",
                    f"{row.get('route_robust_rate', float('nan')):.4f}",
                ]
            )
            + " |"
        )

    best = threshold_df.sort_values(["utility", "f1"], ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Router Threshold",
            "",
            f"- Tuned threshold: `{best['threshold']:.4f}`",
            f"- Validation F1 at threshold: `{best['f1']:.4f}`",
            f"- Route-to-robust rate on validation mixture: `{best['route_robust_rate']:.4f}`",
            "",
            "## Interpretation",
            "",
            "- `normalization_char_tfidf_mlp` tests whether glyph/rune text can be converted back toward Korean before classification.",
            "- `obfuscation_router_char_tfidf_mlp` uses an obfuscation score as a policy: low-score comments use the clean classifier, high-score comments use the robust classifier.",
            "- The router is closer to a deployable moderation pipeline because it does not force every clean comment through a robust model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    norm_model, norm_metrics = train_normalization_defense(args)
    router_metrics, threshold_table = evaluate_router(args)
    metrics_df = pd.concat([norm_metrics, router_metrics], ignore_index=True)

    metrics_path = reports_dir / "defense_model_metrics.csv"
    threshold_path = reports_dir / "router_threshold_search.csv"
    metrics_df.to_csv(metrics_path, index=False)
    threshold_table.to_csv(threshold_path, index=False)
    write_summary(metrics_df, threshold_table, reports_dir / "defense_model_summary.md")
    (reports_dir / "defense_model_run_info.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if args.save_models:
        joblib.dump(norm_model, args.models_dir / "normalization_char_tfidf_mlp.joblib")

    print(metrics_df.to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved summary to {reports_dir / 'defense_model_summary.md'}")


if __name__ == "__main__":
    main()
