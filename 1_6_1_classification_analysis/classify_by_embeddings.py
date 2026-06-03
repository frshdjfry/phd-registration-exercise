#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5

TARGETS = {
    "genre": "first_genre",
    "country_grouped": "country_grouped",
    "decade": "decade",
}

COUNTRY_GROUP_MAP = {
    "United States": "Anglophone",
    "United Kingdom": "Anglophone",
    "Canada": "Anglophone",
    "Australia": "Anglophone",
    "France": "Continental_Europe",
    "Germany": "Continental_Europe",
    "Italy": "Continental_Europe",
    "Spain": "Continental_Europe",
    "Japan": "East_Asia",
    "South Korea": "East_Asia",
    "Hong Kong": "East_Asia",
    "India": "India",
}


def make_output_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_embedding_npz(npz_path: str, prefix: str) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    movie_ids = np.array([str(x) for x in data["movie_ids"]], dtype=object)
    emb = data["embeddings"].astype(np.float32)

    cols = [f"{prefix}{i}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, columns=cols)
    df.insert(0, "movie_id", movie_ids)
    return df


def load_metadata(movies_parquet: str) -> pd.DataFrame:
    movies = pd.read_parquet(movies_parquet).copy()
    movies["movie_id"] = movies["movie_id"].astype(str)
    movies["decade"] = (movies["year"] // 10) * 10
    movies["country_grouped"] = movies["first_country"].map(COUNTRY_GROUP_MAP)

    keep_cols = [
        "movie_id", "title", "year", "decade", "runtime",
        "first_genre", "first_country", "country_grouped"
    ]
    keep_cols = [c for c in keep_cols if c in movies.columns]
    return movies[keep_cols].drop_duplicates("movie_id")


def encode_target(y: pd.Series):
    classes = sorted(pd.unique(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(class_to_idx).astype(int)
    return y_enc, classes


def build_pipeline(num_classes: int, pca_components: int = 0) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if pca_components and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


def evaluate_task(df: pd.DataFrame, target_col: str, feature_cols: list[str], output_prefix: str, pca_components: int):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    nunique = X.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]

    if X.shape[1] == 0:
        raise RuntimeError(f"No usable features for target {target_col}")

    y_enc, classes = encode_target(y)
    num_classes = len(classes)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(num_classes=num_classes, pca_components=pca_components)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    cv_scores = cross_validate(pipe, X, y_enc, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
    y_pred_enc = cross_val_predict(pipe, X, y_enc, cv=cv, n_jobs=1)

    idx_to_class = {i: c for i, c in enumerate(classes)}
    y_pred = pd.Series(y_pred_enc).map(idx_to_class)

    metrics = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(num_classes),
        "pca_components": int(pca_components) if pca_components else 0,
        "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "balanced_accuracy_mean": float(np.mean(cv_scores["test_balanced_accuracy"])),
        "balanced_accuracy_std": float(np.std(cv_scores["test_balanced_accuracy"])),
        "f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
        "f1_macro_std": float(np.std(cv_scores["test_f1_macro"])),
        "accuracy_cv_predict": float(accuracy_score(y_enc, y_pred_enc)),
        "balanced_accuracy_cv_predict": float(balanced_accuracy_score(y_enc, y_pred_enc)),
        "f1_macro_cv_predict": float(f1_score(y_enc, y_pred_enc, average="macro")),
    }

    cm = confusion_matrix(y, y_pred, labels=classes)
    report = classification_report(y, y_pred, output_dict=True)

    with open(f"{output_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"{output_prefix}_confusion_matrix.csv")

    with open(f"{output_prefix}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return metrics


def run_modality_ablation(df: pd.DataFrame, target_col: str, modality_groups: dict[str, list[str]], output_path: str, pca_components: int):
    rows = []

    y = df[target_col].copy()
    valid = y.notna()
    y = y.loc[valid].reset_index(drop=True)
    y_enc, classes = encode_target(y)
    num_classes = len(classes)

    for group_name, cols in modality_groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        X = df.loc[valid, cols].reset_index(drop=True)
        nunique = X.nunique(dropna=False)
        keep_cols = nunique[nunique > 1].index.tolist()
        X = X[keep_cols]

        if X.shape[1] == 0:
            continue

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        pipe = build_pipeline(num_classes=num_classes, pca_components=pca_components)

        scores = cross_validate(
            pipe, X, y_enc, cv=cv,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
            n_jobs=1, return_train_score=False
        )

        rows.append({
            "group": group_name,
            "n_features": int(X.shape[1]),
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
            "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
            "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "f1_macro_std": float(np.std(scores["test_f1_macro"])),
        })

    out_df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    out_df.to_csv(output_path, index=False)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies_parquet", default="movies.parquet")
    ap.add_argument("--dialogue_npz", default='dialogue_film_embeddings.npz', help="Pooled dialogue embeddings .npz")
    ap.add_argument("--music_npz", default='music_film_embeddings.npz', help="Pooled music embeddings .npz")
    ap.add_argument("--image_npz", default='visual_film_embeddings.npz', help="Pooled image embeddings .npz")
    ap.add_argument("--output_dir", default="classification_results_embeddings")
    ap.add_argument("--pca_components", type=int, default=0)
    args = ap.parse_args()

    make_output_dir(args.output_dir)

    meta = load_metadata(args.movies_parquet)
    txt_df = load_embedding_npz(args.dialogue_npz, prefix="txtemb_")
    aud_df = load_embedding_npz(args.music_npz, prefix="audemb_")
    vis_df = load_embedding_npz(args.image_npz, prefix="visemb_")

    df = reduce(lambda left, right: pd.merge(left, right, on="movie_id", how="inner"), [meta, txt_df, aud_df, vis_df])

    feature_cols = [c for c in df.columns if c.startswith(("txtemb_", "audemb_", "visemb_"))]
    modality_groups = {
        "text_embeddings": [c for c in feature_cols if c.startswith("txtemb_")],
        "audio_embeddings": [c for c in feature_cols if c.startswith("audemb_")],
        "image_embeddings": [c for c in feature_cols if c.startswith("visemb_")],
        "all_embeddings": feature_cols,
    }

    summary_rows = []

    for task_name, target_col in TARGETS.items():
        print(f"\n=== Task: {task_name} ===")
        prefix = str(Path(args.output_dir) / task_name)

        metrics = evaluate_task(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            output_prefix=prefix,
            pca_components=args.pca_components,
        )

        ablation_df = run_modality_ablation(
            df=df,
            target_col=target_col,
            modality_groups=modality_groups,
            output_path=f"{prefix}_modality_ablation.csv",
            pca_components=args.pca_components,
        )

        summary_rows.append({"task": task_name, **metrics})

        print("Modality ablation:")
        print(ablation_df.to_string(index=False))

    pd.DataFrame(summary_rows).to_csv(Path(args.output_dir) / "summary_metrics.csv", index=False)

    print("\nSaved all outputs to:", args.output_dir)
    print("Merged films:", df["movie_id"].nunique())


if __name__ == "__main__":
    main()