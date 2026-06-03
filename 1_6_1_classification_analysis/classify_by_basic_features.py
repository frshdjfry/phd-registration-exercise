#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier


INPUT_CSV = "film_level_all_features.csv"
OUTPUT_DIR = "classification_results_basic_features-0"
RANDOM_STATE = 42
N_SPLITS = 5
N_PERM_REPEATS = 20

META_COLS = {
    "movie_id",
    "title",
    "year",
    "runtime",
    "first_genre",
    "first_country",
    "decade",
    "country_grouped",
}

TASKS = {
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


def build_pipeline(num_classes: int) -> Pipeline:
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
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )


def get_all_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = get_all_feature_columns(df)
    selected = []

    for c in cols:
        cl = c.lower()

        if "dialogue_id" in cl:
            continue
        if "vis_luminance_std_" in cl:
            continue

        bad_patterns = [
            "hue_histogram",
            "local_binary_pattern_histogram",
            "flow_direction_histogram",
            "mfcc",
            "chromagram",
        ]
        if any(pat in cl for pat in bad_patterns):
            continue

        selected.append(c)

    return selected


def get_modality_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    groups = {
        "visual_frame": [c for c in feature_cols if c.startswith("vis_")],
        "visual_motion": [c for c in feature_cols if c.startswith("mot_")],
        "audio": [c for c in feature_cols if c.startswith("aud_")],
        "text": [c for c in feature_cols if c.startswith("txt_")],
    }
    groups["visual_all"] = groups["visual_frame"] + groups["visual_motion"]
    groups["all"] = feature_cols[:]
    return groups


def filter_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    return X[keep]


def encode_target(y: pd.Series):
    classes = sorted(pd.unique(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(class_to_idx).astype(int)
    return y_enc, classes


def evaluate_task(df: pd.DataFrame, target_col: str, feature_cols: List[str], output_prefix: str):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    X = filter_constant_columns(X)
    feature_cols = X.columns.tolist()

    if X.shape[1] == 0:
        raise RuntimeError(f"No usable feature columns for target {target_col}")

    y_enc, classes = encode_target(y)
    num_classes = len(classes)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(num_classes=num_classes)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    cv_scores = cross_validate(
        pipe, X, y_enc, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
    )

    y_pred_enc = cross_val_predict(pipe, X, y_enc, cv=cv, n_jobs=1)
    idx_to_class = {i: c for i, c in enumerate(classes)}
    y_pred = pd.Series(y_pred_enc).map(idx_to_class)

    metrics = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(num_classes),
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

    pipe.fit(X, y_enc)
    perm = permutation_importance(
        pipe, X, y_enc, n_repeats=N_PERM_REPEATS,
        random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=1
    )

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    with open(f"{output_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"{output_prefix}_confusion_matrix.csv")

    with open(f"{output_prefix}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    imp_df.to_csv(f"{output_prefix}_permutation_importance.csv", index=False)
    return metrics, imp_df


def run_modality_ablation(df: pd.DataFrame, target_col: str, modality_groups: Dict[str, List[str]], output_path: str):
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
        X = filter_constant_columns(X)
        if X.shape[1] == 0:
            continue

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        pipe = build_pipeline(num_classes=num_classes)

        scores = cross_validate(
            pipe, X, y_enc, cv=cv,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
            n_jobs=1, return_train_score=False
        )

        rows.append({
            "group": group_name,
            "n_features": X.shape[1],
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
    make_output_dir(OUTPUT_DIR)

    df = pd.read_csv(INPUT_CSV)
    df["country_grouped"] = df["first_country"].map(COUNTRY_GROUP_MAP)

    summary_rows = []
    selected_features = select_feature_columns(df)
    modality_groups = get_modality_groups(selected_features)

    print("\n==============================")
    print("Running compact feature set only")
    print(f"Selected features: {len(selected_features)}")
    print("==============================")

    for task_name, target_col in TASKS.items():
        print(f"\n=== Task: {task_name} ===")
        prefix = str(Path(OUTPUT_DIR) / task_name)

        metrics, imp_df = evaluate_task(
            df=df,
            target_col=target_col,
            feature_cols=selected_features,
            output_prefix=prefix,
        )

        ablation_df = run_modality_ablation(
            df=df,
            target_col=target_col,
            modality_groups=modality_groups,
            output_path=f"{prefix}_modality_ablation.csv",
        )

        summary_rows.append({"task": task_name, **metrics})

        print("Top 10 features:")
        print(imp_df.head(10).to_string(index=False))
        print("\nModality ablation:")
        print(ablation_df.to_string(index=False))

    pd.DataFrame(summary_rows).to_csv(Path(OUTPUT_DIR) / "summary_metrics.csv", index=False)

    print("\nSaved all outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()