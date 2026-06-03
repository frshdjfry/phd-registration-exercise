#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce


MOVIES_PARQUET = "movies.parquet"
BASIC_VISUAL_DIR = "basic_visual_features"
BASIC_MOTION_DIR = "basic_motion_features"
MIR_DIR = "mirtoolbox_features"
EEROLA_DIR = "eerola_features"
DIALOGUE_CSV = "dialogues_with_text_features.csv"
OUTPUT_CSV = "film_level_all_features.csv"

TARGET_GENRES = ["Action", "Comedy", "Drama", "Crime", "Horror"]
VALID_DECADES = [2000, 2010, 2020]
AGG_STATS = ["mean", "std"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) != ""]
            new_cols.append("_".join(parts))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def aggregate_numeric_csv_folder(
    folder: str,
    prefix: str,
    exclude_cols: set[str],
    drop_name_patterns: list[str] | None = None,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    stats = stats or AGG_STATS
    drop_name_patterns = drop_name_patterns or []

    rows = []
    for csv_path in sorted(Path(folder).glob("*.csv")):
        movie_id = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}")
            continue

        if df.empty:
            continue

        # drop known metadata/time columns
        existing_exclude = [c for c in exclude_cols if c in df.columns]
        if existing_exclude:
            df = df.drop(columns=existing_exclude)

        # keep numeric columns only
        num_df = df.select_dtypes(include=[np.number]).copy()
        if num_df.empty:
            continue

        # drop unwanted feature families by substring
        keep_cols = []
        for c in num_df.columns:
            if any(pat in c for pat in drop_name_patterns):
                continue
            keep_cols.append(c)
        num_df = num_df[keep_cols]
        if num_df.empty:
            continue

        agg = num_df.agg(stats).T
        row = {"movie_id": str(movie_id)}
        for feat_name, vals in agg.iterrows():
            for stat in stats:
                row[f"{prefix}{feat_name}_{stat}"] = vals[stat]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["movie_id"])
    out["movie_id"] = out["movie_id"].astype(str)
    return out


def aggregate_eerola_only(
    folder: str,
    prefix: str = "aud_",
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep only chroma/key/mode-related numeric columns from Eerola files.
    """
    stats = stats or AGG_STATS
    rows = []

    for csv_path in sorted(Path(folder).glob("*.csv")):
        movie_id = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}")
            continue

        if df.empty:
            continue

        keep_cols = []
        for c in df.columns:
            cl = c.lower()
            if ("chrom" in cl) or ("key" in cl) or ("mode" in cl):
                keep_cols.append(c)

        if not keep_cols:
            continue

        sub = df[keep_cols].copy()
        for c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.select_dtypes(include=[np.number])

        if sub.empty:
            continue

        agg = sub.agg(stats).T
        row = {"movie_id": str(movie_id)}
        for feat_name, vals in agg.iterrows():
            for stat in stats:
                row[f"{prefix}eerola_{feat_name}_{stat}"] = vals[stat]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["movie_id"])
    out["movie_id"] = out["movie_id"].astype(str)
    return out


def aggregate_dialogue_features(
    dialogue_csv: str,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    stats = stats or AGG_STATS
    df = pd.read_csv(dialogue_csv)
    df["movie_id"] = df["movie_id"].astype(str)

    # remove metadata/text/time columns
    drop_cols = [
        "text_clean",
        "text_emotion",
        "first_genre",
        "first_country",
        "decade",
        "runtime",
        "year",
        "start_time",
        "end_time",
        "dialogue_duration",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude coverage indicators if desired
    exclude_text = {
        "sentence_token_count",
        "sentence_token_count_with_lexicon_entry",
    }
    num_cols = [c for c in num_cols if c not in exclude_text]

    grouped = df.groupby("movie_id")[num_cols].agg(stats)
    grouped = flatten_columns(grouped).reset_index()
    grouped = grouped.rename(columns={c: f"txt_{c}" for c in grouped.columns if c != "movie_id"})
    return grouped


def load_metadata(movies_parquet: str) -> pd.DataFrame:
    movies = pd.read_parquet(movies_parquet).copy()
    movies["movie_id"] = movies["movie_id"].astype(str)
    movies["decade"] = (movies["year"] // 10) * 10

    movies = movies[
        movies["first_genre"].isin(TARGET_GENRES)
        & movies["decade"].isin(VALID_DECADES)
        & movies["runtime"].notna()
        & (movies["runtime"] > 0)
    ].copy()

    keep_cols = [
        "movie_id",
        "title",
        "year",
        "decade",
        "runtime",
        "first_genre",
        "first_country",
    ]
    keep_cols = [c for c in keep_cols if c in movies.columns]
    return movies[keep_cols].drop_duplicates("movie_id")


def main():
    meta = load_metadata(MOVIES_PARQUET)

    # 1) basic frame-based visual features
    vis_df = aggregate_numeric_csv_folder(
        folder=BASIC_VISUAL_DIR,
        prefix="vis_",
        exclude_cols={"movie_id", "timestamp"},
        drop_name_patterns=[],
        stats=AGG_STATS,
    )

    # 2) basic motion features
    motion_df = aggregate_numeric_csv_folder(
        folder=BASIC_MOTION_DIR,
        prefix="mot_",
        exclude_cols={
            "segment_index",
            "segment_start_time_seconds",
            "segment_end_time_seconds",
        },
        drop_name_patterns=[],
        stats=AGG_STATS,
    )

    # 3) MIRtoolbox music features, excluding chroma/key/mode
    mir_df = aggregate_numeric_csv_folder(
        folder=MIR_DIR,
        prefix="aud_",
        exclude_cols={
            "movie_id",
            "title",
            "first_genre",
            "cue_index",
            "soundtrack_id",
            "start_time",
            "end_time",
            "embedding_path",
        },
        drop_name_patterns=[
            "chromagram",
            "key",
            "mode",
        ],
        stats=AGG_STATS,
    )

    # 4) Eerola features, only chroma/key/mode
    eerola_df = aggregate_eerola_only(
        folder=EEROLA_DIR,
        prefix="aud_",
        stats=AGG_STATS,
    )

    # 5) dialogue features
    txt_df = aggregate_dialogue_features(DIALOGUE_CSV, stats=AGG_STATS)

    dfs = [meta, vis_df, motion_df, mir_df, eerola_df, txt_df]
    final_df = reduce(lambda left, right: pd.merge(left, right, on="movie_id", how="left"), dfs)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print("Shape:", final_df.shape)
    print("Movies:", final_df['movie_id'].nunique())
    print("Feature columns:", len(final_df.columns) - len(meta.columns))


if __name__ == "__main__":
    main()