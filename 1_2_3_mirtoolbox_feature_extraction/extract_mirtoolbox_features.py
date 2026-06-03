import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pymirtoolbox import feature_extractor
from tqdm import tqdm

FEATURES = [
    ("mirbrightness", "brightness"),
    ("mirtempo", "tempo"),
    ("mirmfcc", "mfcc"),
    ("mirchromagram", "chromagram"),
    ("mirinharmonicity", "inharmonicity"),
    ("mirkey", "key"),
    ("mirroughness", "roughness"),
    ("mireventdensity", "eventdensity"),
    ("mirmode", "mode"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genre", required=True)
    p.add_argument("--cues-folder", default="audio_data")
    p.add_argument("--movies-parquet", default="movies.parquet")
    p.add_argument("--cues-parquet", default="music_cues.parquet")
    p.add_argument("--output-folder", default="mirtoolbox_features")
    return p.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def parse_cue_index_from_embedding_path(embedding_path: str) -> Optional[int]:
    m = re.search(r"_(\d+)\.npy$", str(embedding_path))
    return int(m.group(1)) if m else None


def find_audio_path(cues_folder: Path, movie_id: str, cue_index: int) -> Optional[Path]:
    base_dir = cues_folder / movie_id
    candidates = [
        base_dir / f"soundtrack_{cue_index}.aac",
        base_dir / f"soundtrack_{cue_index}.m4a",
        base_dir / f"soundtrack_{cue_index}.mp4",
        base_dir / f"soundtrack_{cue_index}.wav",
    ]
    for p in candidates:
        if p.exists():
            return p
    if base_dir.exists():
        for p in base_dir.glob(f"soundtrack_{cue_index}.*"):
            if p.is_file():
                return p
    return None


def convert_to_mono_wav(input_audio: Path, tmpdir: Path, out_name: str) -> Path:
    out_wav = tmpdir / out_name
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")
    return out_wav


def expand_feature(prefix: str, value: Any) -> Dict[str, Any]:
    if value is None:
        raise ValueError(f"{prefix} returned None")
    if isinstance(value, str):
        return {prefix: value}
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            v = arr.item()
            return {prefix: v if isinstance(v, str) else str(v)}
        if arr.ndim == 1:
            out: Dict[str, Any] = {}
            for i, v in enumerate(arr, start=1):
                vv = v if isinstance(v, str) else str(v)
                out[f"{prefix}_{i}"] = vv
            return out
        raise RuntimeError(f"{prefix} returned unsupported object array shape {arr.shape}")
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return {prefix: float(arr)}
    if arr.ndim == 1:
        out: Dict[str, Any] = {}
        for i, v in enumerate(arr, start=1):
            out[f"{prefix}_{i}"] = float(v)
        return out
    raise RuntimeError(f"{prefix} returned unexpected numeric shape {arr.shape}")


@contextmanager
def suppress_fds():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)


def extract_features(wav_path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    with suppress_fds():
        for func_name, out_key in FEATURES:
            func = getattr(feature_extractor, func_name)
            result = func(audio_input=str(wav_path))
            if not isinstance(result, dict) or out_key not in result:
                raise RuntimeError(f"{func_name} did not return expected key '{out_key}'")
            val = result[out_key]
            if out_key == "roughness":
                arr = np.asarray(val)
                if arr.dtype == object:
                    raise RuntimeError(f"roughness returned unsupported dtype=object shape={arr.shape}")
                row["roughness"] = float(np.nanmean(arr.astype(np.float64)))
                continue
            row.update(expand_feature(out_key, val))
    return row


def prepare_merged_df(movies_parquet: Path, cues_parquet: Path, genre: str) -> pd.DataFrame:
    movies_df = pd.read_parquet(movies_parquet)
    cues_df = pd.read_parquet(cues_parquet)
    if "movie_id" not in movies_df.columns or "first_genre" not in movies_df.columns:
        raise RuntimeError("movies parquet must include movie_id and first_genre")
    if "movie_id" not in cues_df.columns or "embedding_path" not in cues_df.columns:
        raise RuntimeError("music_cues parquet must include movie_id and embedding_path")
    movies_df = movies_df.copy()
    cues_df = cues_df.copy()
    movies_df["movie_id"] = movies_df["movie_id"].astype(str)
    cues_df["movie_id"] = cues_df["movie_id"].astype(str)
    selected_movies = movies_df[movies_df["first_genre"].astype(str) == genre][
        ["movie_id", "title", "first_genre"]
    ]
    selected_ids = set(selected_movies["movie_id"].tolist())
    if not selected_ids:
        return pd.DataFrame()
    cues_df = cues_df[cues_df["movie_id"].isin(selected_ids)]
    merged = cues_df.merge(selected_movies, on="movie_id", how="left")
    return merged


def load_existing_state(output_folder: Path):
    processed = set()
    for f in output_folder.glob("*.csv"):
        if f.name == "movie_errors.csv":
            continue
        processed.add(f.stem)
    errored = set()
    err_path = output_folder / "movie_errors.csv"
    if err_path.exists():
        try:
            err_df = pd.read_csv(err_path, dtype={"movie_id": str})
            if "movie_id" in err_df.columns:
                errored.update(err_df["movie_id"].astype(str).tolist())
        except Exception:
            pass
    return processed, errored


def append_movie_error(output_folder: Path, movie_id: str, title: str, first_genre: str, error_msg: str) -> None:
    err_path = output_folder / "movie_errors.csv"
    row_df = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "title": title,
                "first_genre": first_genre,
                "error": error_msg,
            }
        ]
    )
    if err_path.exists():
        row_df.to_csv(err_path, mode="a", index=False, header=False)
    else:
        row_df.to_csv(err_path, index=False)


def compute_remaining_cues(merged: pd.DataFrame, processed_movies: set, errored_movies: set) -> int:
    if merged.empty:
        return 0
    mask = ~merged["movie_id"].astype(str).isin(processed_movies | errored_movies)
    return int(mask.sum())


def build_movie_features_df(
    movie_id: str,
    title: str,
    first_genre: str,
    group: pd.DataFrame,
    cues_folder: Path,
    tmpdir: Path,
    pbar,
) -> pd.DataFrame:
    rows = []
    expected_feature_keys = None
    for _, cue in group.iterrows():
        try:
            embedding_path = cue.get("embedding_path", "")
            cue_index = parse_cue_index_from_embedding_path(embedding_path)
            if cue_index is None:
                raise RuntimeError(f"could not parse cue index from embedding_path='{embedding_path}'")
            audio_path = find_audio_path(cues_folder, movie_id, cue_index)
            if audio_path is None:
                raise RuntimeError(f"audio file not found for movie_id={movie_id} cue_index={cue_index}")
            wav_name = f"{movie_id}_soundtrack_{cue_index}.wav"
            wav_path = convert_to_mono_wav(audio_path, tmpdir, wav_name)
            feats = extract_features(wav_path)
            feature_keys = tuple(sorted(feats.keys()))
            if expected_feature_keys is None:
                expected_feature_keys = feature_keys
            elif feature_keys != expected_feature_keys:
                raise RuntimeError("inconsistent feature keys across cues for this movie")
            base = {
                "movie_id": movie_id,
                "title": title,
                "first_genre": first_genre,
                "cue_index": cue_index,
                "soundtrack_id": cue.get("soundtrack_id", None),
                "start_time": cue.get("start_time", None),
                "end_time": cue.get("end_time", None),
                "embedding_path": embedding_path,
            }
            base.update(feats)
            rows.append(base)
        finally:
            if pbar is not None:
                pbar.update(1)
    if not rows:
        raise RuntimeError("no cues for this movie")
    return pd.DataFrame(rows)


def process_all_movies(merged: pd.DataFrame, cues_folder: Path, output_folder: Path) -> None:
    processed_movies, errored_movies = load_existing_state(output_folder)
    total_cues = compute_remaining_cues(merged, processed_movies, errored_movies)
    if total_cues <= 0:
        return
    with tempfile.TemporaryDirectory(prefix="mir_tmp_") as td:
        tmpdir = Path(td)
        with tqdm(total=total_cues, unit="cue", dynamic_ncols=True) as pbar:
            for movie_id, group in merged.groupby("movie_id", sort=True):
                movie_id_str = str(movie_id)
                if movie_id_str in processed_movies or movie_id_str in errored_movies:
                    continue
                title = ""
                if "title" in group.columns and len(group["title"]) > 0 and pd.notna(group["title"].iloc[0]):
                    title = str(group["title"].iloc[0])
                first_genre = ""
                if "first_genre" in group.columns and len(group["first_genre"]) > 0 and pd.notna(
                    group["first_genre"].iloc[0]
                ):
                    first_genre = str(group["first_genre"].iloc[0])
                pbar.set_description(f"movie {movie_id_str}")
                try:
                    df = build_movie_features_df(movie_id_str, title, first_genre, group, cues_folder, tmpdir, pbar)
                except Exception as e:
                    append_movie_error(
                        output_folder,
                        movie_id_str,
                        title,
                        first_genre,
                        f"{type(e).__name__}: {e}",
                    )
                    errored_movies.add(movie_id_str)
                    continue
                out_path = output_folder / f"{movie_id_str}.csv"
                df.to_csv(out_path, index=False)
                processed_movies.add(movie_id_str)


def run():
    args = parse_args()
    require_ffmpeg()
    cues_folder = Path(args.cues_folder)
    movies_parquet = Path(args.movies_parquet)
    cues_parquet = Path(args.cues_parquet)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    merged = prepare_merged_df(movies_parquet, cues_parquet, args.genre)
    if merged.empty:
        return
    process_all_movies(merged, cues_folder, output_folder)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
