#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DPI = 220
FIG_EXTS = ["png", "pdf"]

METRICS = [
    ("accuracy_mean", "accuracy_std", "Accuracy"),
    ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy"),
    ("f1_macro_mean", "f1_macro_std", "Macro-F1"),
]

TASK_LABELS = {
    "genre": "Genre",
    "decade": "Decade",
    "country_grouped": "Grouped Countries",
}

GROUP_LABELS = {
    "all": "All Modalities",
    "visual_all": "Visual",
    "audio": "Music",
    "text": "Dialogue",
    "all_embeddings": "All Modalities",
    "image_embeddings": "Visual",
    "audio_embeddings": "Music",
    "text_embeddings": "Dialogue",
}


def set_clean_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.8),
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTS:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_result_bundle(result_dir: Path):
    tasks = {}
    for task_key in TASK_LABELS.keys():
        metrics_path = result_dir / f"{task_key}_metrics.json"
        cm_path = result_dir / f"{task_key}_confusion_matrix.csv"
        ablation_path = result_dir / f"{task_key}_modality_ablation.csv"

        bundle = {}
        if metrics_path.exists():
            bundle["metrics"] = load_json(metrics_path)
        if cm_path.exists():
            bundle["confusion"] = pd.read_csv(cm_path, index_col=0)
        if ablation_path.exists():
            bundle["ablation"] = pd.read_csv(ablation_path)

        if bundle:
            tasks[task_key] = bundle
    return tasks


def prettify_group_names(ab_df: pd.DataFrame) -> pd.DataFrame:
    ab_df = ab_df.copy()
    ab_df["group_pretty"] = ab_df["group"].map(lambda x: GROUP_LABELS.get(x, x))
    return ab_df


def add_vertical_bar_labels(ax, bars, values, errors=None, fmt="{:.2f}", fontsize=8):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    for i, (bar, val) in enumerate(zip(bars, values)):
        err = 0.0 if errors is None else float(errors[i])
        y = val + err + 0.015 * yr
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def add_horizontal_bar_labels(ax, bars, values, errors=None, fmt="{:.2f}", fontsize=8):
    xmin, xmax = ax.get_xlim()
    xr = xmax - xmin
    for i, (bar, val) in enumerate(zip(bars, values)):
        err = 0.0 if errors is None else float(errors[i])
        x = val + err + 0.01 * xr
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            x,
            y,
            fmt.format(val),
            ha="left",
            va="center",
            fontsize=fontsize,
        )


def plot_overall_comparison_side_by_side(basic_tasks: dict, emb_tasks: dict, out_base: Path):
    task_keys = ["genre", "decade", "country_grouped"]
    x = np.arange(len(task_keys))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    model_specs = [
        ("Handcrafted Features", basic_tasks, axes[0]),
        ("Embeddings", emb_tasks, axes[1]),
    ]

    for title, tasks, ax in model_specs:
        for i, (mean_key, std_key, metric_label) in enumerate(METRICS):
            means = [tasks[t]["metrics"][mean_key] for t in task_keys]
            stds = [tasks[t]["metrics"][std_key] for t in task_keys]

            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                alpha=0.9,
                label=metric_label,
            )
            add_vertical_bar_labels(ax, bars, means, errors=stds, fmt="{:.2f}", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in task_keys])
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Score")
    fig.suptitle("Classification Performance Across Tasks", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(fig, out_base)


def plot_confusion_matrix(ax, cm_df: pd.DataFrame, panel_title: str):
    cm = cm_df.values.astype(float)
    labels = [pretty_class_label(x) for x in cm_df.index.tolist()]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    im = ax.imshow(cm_norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(panel_title)

    threshold = cm_norm.max() * 0.5 if cm_norm.size else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            frac = cm_norm[i, j]
            raw = int(cm[i, j])
            color = "black" if frac >= threshold else "white"
            ax.text(
                j, i,
                f"{raw}\n{frac:.2f}",
                ha="center", va="center",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("")
    ax.set_ylabel("")
    return im

def pretty_class_label(x: str) -> str:
    mapping = {
        "Continental_Europe": "Continental Europe",
        "East_Asia": "East Asia",
        "India": "South Asia",
    }
    return mapping.get(str(x), str(x))

def sort_ablation_df(ab_df: pd.DataFrame) -> pd.DataFrame:
    ab_df = prettify_group_names(ab_df)

    # remove separate image/motion rows for handcrafted plots
    ab_df = ab_df[~ab_df["group"].isin(["visual_frame", "visual_motion"])].copy()

    preferred_order = [
        "Dialogue",
        "Music",
        "Visual",
        "All Modalities",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}
    ab_df["sort_rank"] = ab_df["group_pretty"].map(lambda x: rank.get(x, 999))
    ab_df = ab_df.sort_values(["sort_rank", "group_pretty"])
    return ab_df

def plot_ablation(ax, ab_df: pd.DataFrame, panel_title: str):
    ab_df = sort_ablation_df(ab_df)

    y = np.arange(len(ab_df))
    means = ab_df["f1_macro_mean"].to_numpy()
    stds = ab_df["f1_macro_std"].to_numpy()

    bars = ax.barh(
        y,
        means,
        xerr=stds,
        capsize=3,
        alpha=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ab_df["group_pretty"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.set_title(panel_title)
    ax.grid(axis="x", alpha=0.2)

    add_horizontal_bar_labels(ax, bars, means, errors=stds, fmt="{:.2f}", fontsize=8)


def make_confusion_figure(task_key: str, basic_bundle: dict, emb_bundle: dict, out_base: Path):
    task_label = TASK_LABELS[task_key]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    im1 = plot_confusion_matrix(
        axes[0],
        basic_bundle["confusion"],
        "Handcrafted Features",
    )
    im2 = plot_confusion_matrix(
        axes[1],
        emb_bundle["confusion"],
        "Embeddings",
    )

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    # cbar1.set_label("Row-Normalized Proportion")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    # cbar2.set_label("Row-Normalized Proportion")

    fig.suptitle(f"{task_label} Confusion Matrices", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, out_base)


def make_ablation_figure(task_key: str, basic_bundle: dict, emb_bundle: dict, out_base: Path):
    task_label = TASK_LABELS[task_key]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)

    plot_ablation(
        axes[0],
        basic_bundle["ablation"],
        "Handcrafted Features",
    )
    plot_ablation(
        axes[1],
        emb_bundle["ablation"],
        "Embeddings",
    )

    fig.suptitle(f"{task_label} Modality Ablations", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic_results_dir", default='classification_results_basic_features')
    ap.add_argument("--embedding_results_dir", default='classification_results_embeddings')
    ap.add_argument("--out_dir", default="classification_report_plots")
    args = ap.parse_args()

    set_clean_style()

    basic_dir = Path(args.basic_results_dir)
    emb_dir = Path(args.embedding_results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basic_tasks = load_result_bundle(basic_dir)
    emb_tasks = load_result_bundle(emb_dir)

    required = ["genre", "decade", "country_grouped"]
    for task in required:
        if task not in basic_tasks:
            raise RuntimeError(f"Missing basic-results files for task: {task}")
        if task not in emb_tasks:
            raise RuntimeError(f"Missing embedding-results files for task: {task}")

    plot_overall_comparison_side_by_side(
        basic_tasks,
        emb_tasks,
        out_dir / "overall_comparison_side_by_side",
    )

    for task in required:
        make_confusion_figure(
            task_key=task,
            basic_bundle=basic_tasks[task],
            emb_bundle=emb_tasks[task],
            out_base=out_dir / f"{task}_confusion_matrices",
        )
        make_ablation_figure(
            task_key=task,
            basic_bundle=basic_tasks[task],
            emb_bundle=emb_tasks[task],
            out_base=out_dir / f"{task}_modality_ablations",
        )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()