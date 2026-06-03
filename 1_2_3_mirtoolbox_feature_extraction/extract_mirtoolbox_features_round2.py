import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymirtoolbox import feature_extractor
from tqdm import tqdm


# func_name, output_key
FEATURES = [
    ("mirbeatspectrum", "beatspectrum"),
    ("mirpitch", "pitch"),
    ("mirfluctuation", "fluctuation"),
    ("mirmetroid", "metrical_centroid"),
    ("mirpulseclarity", "pulseclarity"),
    ("mirregularity", "irregularity"),
    ("mirrolloff", "rolloff"),
    ("mirzerocross", "zerocross"),
    ("mirhcdf", "hcdf"),
    ("mirkeystrength", "keystrength"),
    ("mirtonalcentroid", "tonalcentroid"),
]


# Extra keyword arguments for specific MIRToolbox calls.
# mirpitch Total=1 should return one value.
FEATURE_KWARGS = {
    "mirpitch": {"Total": 1},
}


# Features that should be stored as one string cell, not expanded into columns.
ARRAY_AS_STRING_FEATURES = {
    "beatspectrum",
    "fluctuation",
    "metrical_centroid",
    "hcdf",
    "keystrength",
    "tonalcentroid",
}


# Features expected to be scalar floats.
SCALAR_FEATURES = {
    "pitch",
    "pulseclarity",
    "irregularity",
    "rolloff",
    "zerocross",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genre", required=True)
    p.add_argument("--cues-folder", default="/home/fj303@cam.ac.uk/fusic_data")
    p.add_argument("--movies-parquet", default="movies.parquet")
    p.add_argument("--cues-parquet", default="music_cues.parquet")
    p.add_argument("--output-folder", default="mirtoolbox_features_round2")

    # Memory-safe beat spectrum options.
    p.add_argument(
        "--beatspectrum-full-limit",
        type=float,
        default=90.0,
        help="Run mirbeatspectrum normally only up to this cue duration in seconds.",
    )
    p.add_argument(
        "--beatspectrum-chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk duration in seconds for long cues.",
    )
    p.add_argument(
        "--beatspectrum-max-chunks",
        type=int,
        default=5,
        help="Maximum number of evenly spaced chunks to use for long cues.",
    )

    return p.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH")


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


def get_audio_duration_seconds(audio_path: Path) -> Optional[float]:
    """
    Return audio duration using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        return None

    try:
        return float(proc.stdout.strip())
    except Exception:
        return None


def make_chunk_wavs(
    input_wav: Path,
    tmpdir: Path,
    chunk_duration: float,
    max_chunks: int,
) -> List[Path]:
    """
    Create up to max_chunks evenly spaced chunks from a WAV.

    For short files, returns the original WAV path.
    """
    total_duration = get_audio_duration_seconds(input_wav)

    if total_duration is None:
        return [input_wav]

    if total_duration <= chunk_duration:
        return [input_wav]

    if max_chunks <= 1:
        starts = [max(0.0, (total_duration - chunk_duration) / 2.0)]
    else:
        starts = np.linspace(
            0.0,
            max(0.0, total_duration - chunk_duration),
            num=max_chunks,
        )

    chunk_paths: List[Path] = []

    for i, start in enumerate(starts, start=1):
        out_wav = tmpdir / f"{input_wav.stem}_beatspectrum_chunk_{i}.wav"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(float(start)),
            "-i",
            str(input_wav),
            "-t",
            str(float(chunk_duration)),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(out_wav),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0 or not out_wav.exists():
            raise RuntimeError(f"ffmpeg chunk failed: {proc.stderr[-2000:]}")

        chunk_paths.append(out_wav)

    return chunk_paths


def clean_for_json(x: Any, ndigits: int = 6) -> Any:
    """
    Convert numpy / Python values into JSON-safe values.

    Floats are rounded to reduce CSV size.
    NaN / inf are converted to None because strict JSON does not support them.
    """
    if isinstance(x, np.ndarray):
        return clean_for_json(x.tolist(), ndigits=ndigits)

    if isinstance(x, np.generic):
        return clean_for_json(x.item(), ndigits=ndigits)

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, ndigits)

    if isinstance(x, int):
        return x

    if isinstance(x, str):
        return x

    if isinstance(x, bool):
        return x

    if x is None:
        return None

    if isinstance(x, list):
        return [clean_for_json(v, ndigits=ndigits) for v in x]

    if isinstance(x, tuple):
        return [clean_for_json(v, ndigits=ndigits) for v in x]

    if isinstance(x, dict):
        return {str(k): clean_for_json(v, ndigits=ndigits) for k, v in x.items()}

    return str(x)


def array_to_string(value: Any, ndigits: int = 6) -> str:
    """
    Store arrays / matrices in a single table cell as a compact JSON string.

    Singleton dimensions are removed so shapes like:
        (1, N) -> (N,)
        (N, 1) -> (N,)
        (12, 1, 1, 2) -> (12, 2)

    Float values are rounded to reduce CSV size.
    """
    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.dtype == object:
        cleaned = clean_for_json(arr, ndigits=ndigits)
    else:
        cleaned = clean_for_json(arr.astype(np.float64, copy=False), ndigits=ndigits)

    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def scalar_to_float(prefix: str, value: Any, ndigits: int = 6) -> float:
    """
    Convert MIRToolbox scalar-like output to a rounded float.

    If a feature unexpectedly returns an array, we use nanmean so that the script
    can continue while still producing one scalar value.
    """
    arr = np.asarray(value)

    if arr.dtype == object:
        arr = np.squeeze(arr)

        if arr.ndim == 0:
            item = arr.item()
            try:
                return round(float(item), ndigits)
            except Exception as e:
                raise RuntimeError(f"{prefix} returned non-numeric scalar value {item!r}") from e

        raise RuntimeError(f"{prefix} expected scalar but returned object array shape={arr.shape}")

    arr = np.squeeze(arr).astype(np.float64)

    if arr.ndim == 0:
        return round(float(arr), ndigits)

    return round(float(np.nanmean(arr)), ndigits)


def expand_feature(prefix: str, value: Any) -> Dict[str, Any]:
    """
    Convert one MIRToolbox feature into one or more dataframe columns.

    Special cases:
      - variable-length arrays/matrices are stored as strings in one column
      - scalar features are stored as floats
      - unknown 1D numeric arrays are expanded into multiple columns as before
    """
    if value is None:
        raise ValueError(f"{prefix} returned None")

    if prefix in ARRAY_AS_STRING_FEATURES:
        return {prefix: array_to_string(value)}

    if prefix in SCALAR_FEATURES:
        return {prefix: scalar_to_float(prefix, value)}

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
                out[f"{prefix}_{i}"] = v if isinstance(v, str) else str(v)
            return out

        return {prefix: array_to_string(arr)}

    arr = np.squeeze(arr).astype(np.float64)

    if arr.ndim == 0:
        return {prefix: float(arr)}

    if arr.ndim == 1:
        out: Dict[str, Any] = {}
        for i, v in enumerate(arr, start=1):
            out[f"{prefix}_{i}"] = float(v)
        return out

    return {prefix: array_to_string(arr)}


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


def call_feature_function(func_name: str, wav_path: Path) -> Dict[str, Any]:
    """
    Call pymirtoolbox feature function.

    Some functions need special arguments, for example:
        mirpitch(audio_input=..., Total=1)

    If a function rejects the special kwargs, this raises a clear error.
    """
    func = getattr(feature_extractor, func_name)
    kwargs = FEATURE_KWARGS.get(func_name, {})

    try:
        return func(audio_input=str(wav_path), **kwargs)
    except TypeError as e:
        if kwargs:
            raise RuntimeError(
                f"{func_name} failed with kwargs={kwargs}. "
                f"Check whether this pymirtoolbox version supports those arguments."
            ) from e
        raise


def compute_beatspectrum_full(wav_path: Path) -> Any:
    result = call_feature_function("mirbeatspectrum", wav_path)

    if not isinstance(result, dict) or "beatspectrum" not in result:
        raise RuntimeError("mirbeatspectrum did not return expected key 'beatspectrum'")

    return result["beatspectrum"]


def compute_beatspectrum_chunked(
    wav_path: Path,
    tmpdir: Path,
    chunk_seconds: float,
    max_chunks: int,
) -> Any:
    """
    Memory-safe beat spectrum extraction.

    Runs mirbeatspectrum on short chunks, then averages the returned arrays.
    This avoids the huge RAM usage caused by running mirbeatspectrum on a long cue.
    """
    chunk_paths = make_chunk_wavs(
        input_wav=wav_path,
        tmpdir=tmpdir,
        chunk_duration=chunk_seconds,
        max_chunks=max_chunks,
    )

    arrays: List[np.ndarray] = []

    for chunk_path in chunk_paths:
        result = call_feature_function("mirbeatspectrum", chunk_path)

        if not isinstance(result, dict) or "beatspectrum" not in result:
            raise RuntimeError("mirbeatspectrum did not return expected key 'beatspectrum'")

        arr = np.asarray(result["beatspectrum"], dtype=np.float64)
        arr = np.squeeze(arr)

        if arr.ndim == 0:
            arr = arr.reshape(1)

        if arr.ndim != 1:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise RuntimeError(f"mirbeatspectrum returned empty array for chunk {chunk_path}")

        arrays.append(arr.copy())

        del result
        del arr
        gc.collect()

    if not arrays:
        raise RuntimeError("no beat spectrum chunks were computed")

    min_len = min(len(a) for a in arrays)

    if min_len <= 0:
        raise RuntimeError("beat spectrum chunks had invalid lengths")

    arrays = [a[:min_len] for a in arrays]
    mean_arr = np.nanmean(np.vstack(arrays), axis=0)

    return mean_arr


def compute_beatspectrum_safe(
    wav_path: Path,
    tmpdir: Path,
    full_limit_seconds: float,
    chunk_seconds: float,
    max_chunks: int,
) -> Any:
    """
    Run mirbeatspectrum normally for short cues and chunked for long cues.
    """
    duration = get_audio_duration_seconds(wav_path)

    if duration is None:
        # Conservative choice: if duration is unknown, use chunked mode.
        return compute_beatspectrum_chunked(
            wav_path=wav_path,
            tmpdir=tmpdir,
            chunk_seconds=chunk_seconds,
            max_chunks=max_chunks,
        )

    if duration <= full_limit_seconds:
        return compute_beatspectrum_full(wav_path)

    return compute_beatspectrum_chunked(
        wav_path=wav_path,
        tmpdir=tmpdir,
        chunk_seconds=chunk_seconds,
        max_chunks=max_chunks,
    )


def extract_features(
    wav_path: Path,
    tmpdir: Path,
    beatspectrum_full_limit: float,
    beatspectrum_chunk_seconds: float,
    beatspectrum_max_chunks: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    with suppress_fds():
        for func_name, out_key in FEATURES:
            if func_name == "mirbeatspectrum":
                val = compute_beatspectrum_safe(
                    wav_path=wav_path,
                    tmpdir=tmpdir,
                    full_limit_seconds=beatspectrum_full_limit,
                    chunk_seconds=beatspectrum_chunk_seconds,
                    max_chunks=beatspectrum_max_chunks,
                )
                row.update(expand_feature(out_key, val))
                continue

            result = call_feature_function(func_name, wav_path)

            if not isinstance(result, dict) or out_key not in result:
                raise RuntimeError(f"{func_name} did not return expected key '{out_key}'")

            val = result[out_key]
            row.update(expand_feature(out_key, val))

            del result
            gc.collect()

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


def append_movie_error(
    output_folder: Path,
    movie_id: str,
    title: str,
    first_genre: str,
    error_msg: str,
) -> None:
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


def compute_remaining_cues(
    merged: pd.DataFrame,
    processed_movies: set,
    errored_movies: set,
) -> int:
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
    beatspectrum_full_limit: float,
    beatspectrum_chunk_seconds: float,
    beatspectrum_max_chunks: int,
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

            feats = extract_features(
                wav_path=wav_path,
                tmpdir=tmpdir,
                beatspectrum_full_limit=beatspectrum_full_limit,
                beatspectrum_chunk_seconds=beatspectrum_chunk_seconds,
                beatspectrum_max_chunks=beatspectrum_max_chunks,
            )

            feature_keys = tuple(sorted(feats.keys()))

            if expected_feature_keys is None:
                expected_feature_keys = feature_keys
            elif feature_keys != expected_feature_keys:
                raise RuntimeError(
                    "inconsistent feature keys across cues for this movie. "
                    f"Expected {expected_feature_keys}, got {feature_keys}"
                )

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

            del feats
            gc.collect()

        finally:
            if pbar is not None:
                pbar.update(1)

    if not rows:
        raise RuntimeError("no cues for this movie")

    return pd.DataFrame(rows)


def process_all_movies(
    merged: pd.DataFrame,
    cues_folder: Path,
    output_folder: Path,
    beatspectrum_full_limit: float,
    beatspectrum_chunk_seconds: float,
    beatspectrum_max_chunks: int,
) -> None:
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
                if (
                    "first_genre" in group.columns
                    and len(group["first_genre"]) > 0
                    and pd.notna(group["first_genre"].iloc[0])
                ):
                    first_genre = str(group["first_genre"].iloc[0])

                pbar.set_description(f"movie {movie_id_str}")

                try:
                    df = build_movie_features_df(
                        movie_id=movie_id_str,
                        title=title,
                        first_genre=first_genre,
                        group=group,
                        cues_folder=cues_folder,
                        tmpdir=tmpdir,
                        pbar=pbar,
                        beatspectrum_full_limit=beatspectrum_full_limit,
                        beatspectrum_chunk_seconds=beatspectrum_chunk_seconds,
                        beatspectrum_max_chunks=beatspectrum_max_chunks,
                    )
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

                del df
                gc.collect()


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

    process_all_movies(
        merged=merged,
        cues_folder=cues_folder,
        output_folder=output_folder,
        beatspectrum_full_limit=args.beatspectrum_full_limit,
        beatspectrum_chunk_seconds=args.beatspectrum_chunk_seconds,
        beatspectrum_max_chunks=args.beatspectrum_max_chunks,
    )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)