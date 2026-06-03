#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DPI = 220
FIG_EXTS = ["png", "pdf"]

TASK_LABELS = {
    "genre": "Genre",
    "decade": "Decade",
    "country_grouped": "Grouped Countries",
}

PREFIX_LABELS = {
    "txt": "Dialogue",
    "aud": "Music",
    "vis": "Visual",
    "mot": "Motion",
}


def set_clean_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTS:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def prettify_feature_name(name: str) -> str:
    parts = name.split("_")
    if not parts:
        return name

    prefix = parts[0]
    stem = "_".join(parts[1:])

    source = PREFIX_LABELS.get(prefix, prefix)

    # remove common aggregated suffixes
    agg = None
    if stem.endswith("_mean_mean"):
        stem = stem[:-10]
        agg = "mean"
    elif stem.endswith("_mean_std"):
        stem = stem[:-9]
        agg = "std"
    elif stem.endswith("_std_mean"):
        stem = stem[:-9]
        agg = "mean"
    elif stem.endswith("_std_std"):
        stem = stem[:-8]
        agg = "std"
    elif stem.endswith("_mean"):
        stem = stem[:-5]
        agg = "mean"
    elif stem.endswith("_std"):
        stem = stem[:-4]
        agg = "std"

    replacements = [
        ("lexical_decision_reaction_time", "lexical decision time"),
        ("word_frequency_log10", "word frequency"),
        ("lexical_density", "lexical density"),
        ("concreteness_rating", "concreteness"),
        ("imageability_rating", "imageability"),
        ("valence_rating", "valence"),
        ("arousal_rating", "arousal"),
        ("dominance_rating", "dominance"),
        ("sensory_visual_strength", "visual sensory strength"),
        ("sensory_auditory_strength", "auditory sensory strength"),
        ("sensory_haptic_strength", "haptic sensory strength"),
        ("sensory_gustatory_strength", "gustatory sensory strength"),
        ("sensory_olfactory_strength", "olfactory sensory strength"),
        ("sensory_interoceptive_strength", "interoceptive sensory strength"),
        ("flow_magnitude_mean_fraction_of_diagonal_per_second", "flow magnitude"),
        ("flow_magnitude_p95_fraction_of_diagonal_per_second", "flow magnitude (p95)"),
        ("motion_pixel_ratio", "motion pixel ratio"),
        ("camera_translation_x_fraction_of_diagonal_per_second", "camera translation x"),
        ("camera_translation_y_fraction_of_diagonal_per_second", "camera translation y"),
        ("camera_scale_change_per_second", "camera scale change"),
        ("local_binary_pattern_entropy", "LBP entropy"),
        ("edge_energy", "edge energy"),
        ("edge_density", "edge density"),
        ("luminance_mean", "luminance"),
        ("luminance_contrast", "luminance contrast"),
        ("saturation_mean", "saturation"),
        ("colorfulness", "colorfulness"),
        ("warm_cool_ratio", "warm–cool ratio"),
        ("brightness", "brightness"),
        ("tempo", "tempo"),
        ("roughness", "roughness"),
        ("inharmonicity", "inharmonicity"),
        ("eventdensity", "event density"),
        ("eerola_mode", "mode"),
        ("eerola_key", "key"),
    ]

    pretty = stem
    for old, new in replacements:
        pretty = pretty.replace(old, new)

    # chroma bins
    pretty = re.sub(r"eerola_chromagram_(\d+)", r"chromagram bin \1", pretty)

    # mfccs if they appear
    pretty = re.sub(r"mfcc_(\d+)", r"MFCC \1", pretty)

    # hist bins if they appear
    pretty = re.sub(r"hue_histogram_(\d+)", r"hue bin \1", pretty)
    pretty = re.sub(r"flow_direction_histogram_bin_(\d+)", r"flow direction bin \1", pretty)
    pretty = re.sub(r"local_binary_pattern_histogram_(\d+)", r"LBP bin \1", pretty)

    pretty = pretty.replace("_", " ").strip()

    # compact aggregation note
    if agg == "mean":
        pretty = f"{source} {pretty} (mean)"
    elif agg == "std":
        pretty = f"{source} {pretty} (std)"
    else:
        pretty = f"{source} {pretty}"

    # clean duplicate spaces
    pretty = re.sub(r"\s+", " ", pretty)
    return pretty


def plot_importance(imp_df: pd.DataFrame, task_key: str, out_base: Path, top_k: int = 15):
    imp_df = imp_df.sort_values("importance_mean", ascending=False).head(top_k).copy()
    imp_df["feature_clean"] = imp_df["feature"].apply(prettify_feature_name)
    imp_df = imp_df.iloc[::-1]

    fig_h = max(4.8, 0.38 * len(imp_df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))

    y = np.arange(len(imp_df))
    bars = ax.barh(
        y,
        imp_df["importance_mean"].to_numpy(),
        xerr=imp_df["importance_std"].to_numpy(),
        capsize=3,
        alpha=0.9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(imp_df["feature_clean"])
    ax.set_xlabel("Permutation Importance (Macro-F1 Drop)")
    ax.set_title(f"{TASK_LABELS.get(task_key, task_key)}: Top Feature Importances")
    ax.grid(axis="x", alpha=0.2)

    xmin, xmax = ax.get_xlim()
    xr = xmax - xmin
    for bar, mean, std in zip(bars, imp_df["importance_mean"], imp_df["importance_std"]):
        ax.text(
            mean + std + 0.01 * xr,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    save_figure(fig, out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        default="classification_results_basic_features",
        help="Directory containing *_permutation_importance.csv files",
    )
    ap.add_argument(
        "--out_dir",
        default="feature_importance_plots",
        help="Directory to save cleaned plots",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top features to plot",
    )
    ap.add_argument(
        "--task",
        default="all",
        choices=["all", "genre", "decade", "country_grouped"],
        help="Plot one task or all",
    )
    args = ap.parse_args()

    set_clean_style()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_keys = ["genre", "decade", "country_grouped"] if args.task == "all" else [args.task]

    for task_key in task_keys:
        csv_path = results_dir / f"{task_key}_permutation_importance.csv"
        if not csv_path.exists():
            print(f"Skipping {task_key}: file not found -> {csv_path}")
            continue

        imp_df = pd.read_csv(csv_path)
        if imp_df.empty:
            print(f"Skipping {task_key}: empty file")
            continue

        plot_importance(
            imp_df=imp_df,
            task_key=task_key,
            out_base=out_dir / f"{task_key}_feature_importance_clean",
            top_k=args.top_k,
        )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()