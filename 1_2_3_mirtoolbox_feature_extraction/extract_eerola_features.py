import argparse
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


KEY_TO_INT = {
    "C": 1,
    "C#": 2,
    "DB": 2,
    "D": 3,
    "D#": 4,
    "EB": 4,
    "E": 5,
    "F": 6,
    "F#": 7,
    "GB": 7,
    "G": 8,
    "G#": 9,
    "AB": 9,
    "A": 10,
    "A#": 11,
    "BB": 11,
    "B": 12,
    "CB": 12,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genre", required=True)
    p.add_argument("--cues-folder", default="audio_data")
    p.add_argument("--movies-parquet", default="movies.parquet")
    p.add_argument("--cues-parquet", default="music_cues.parquet")
    p.add_argument("--output-folder", default="eerola_features")
    return p.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def load_relative_mode_api():
    errors = []
    for module_name in ("relative_mode", "src.relative_mode"):
        try:
            mod = importlib.import_module(module_name)
            tonal_fragment_cls = getattr(mod, "Tonal_Fragment", None)
            relative_mode_fn = getattr(mod, "relative_mode", None)
            if tonal_fragment_cls is None or relative_mode_fn is None:
                errors.append(f"{module_name}: missing Tonal_Fragment or relative_mode")
                continue
            return tonal_fragment_cls, relative_mode_fn, module_name
        except Exception as e:
            errors.append(f"{module_name}: {type(e).__name__}: {e}")

    raise ImportError(
        "Could not import relative_mode. Install the package/repo first. "
        + " | ".join(errors)
    )


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


def tune_audio_like_relative_mode(y: np.ndarray, sr: int, remove_percussive: bool = False) -> np.ndarray:
    t = librosa.estimate_tuning(y=y, sr=sr)
    y440 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-t)

    if remove_percussive:
        y440_stft = librosa.stft(y440)
        y440_stft_harmonic, _ = librosa.decompose.hpss(y440_stft)
        y440 = librosa.istft(y440_stft_harmonic, length=len(y440))

    return y440


def extract_cqt_chromagram(y440: np.ndarray, sr: int) -> Dict[str, float]:
    chroma = librosa.feature.chroma_cqt(
        y=y440,
        sr=sr,
        n_octaves=7,
        threshold=0.0,
        fmin=65.4,
        bins_per_octave=36,
        cqt_mode="hybrid",
        hop_length=8192,
    )

    # Aggregate to one 12-bin vector, then normalize so max == 1.0 like the old CSV style
    chroma_vec = np.sum(chroma, axis=1).astype(np.float64)
    max_val = float(np.max(chroma_vec)) if chroma_vec.size else 0.0
    if max_val > 0.0:
        chroma_vec = chroma_vec / max_val

    if chroma_vec.shape != (12,):
        raise RuntimeError(f"unexpected chromagram shape after aggregation: {chroma_vec.shape}")

    return {f"chromagram_{i+1}": float(chroma_vec[i]) for i in range(12)}


def tonic_label_to_int(key_label: str) -> int:
    tonic = str(key_label).strip().split()[0].upper()
    tonic = tonic.replace("♯", "#").replace("♭", "B")
    if tonic not in KEY_TO_INT:
        raise RuntimeError(f"unsupported key label: {key_label}")
    return int(KEY_TO_INT[tonic])


def extract_key_int(y440: np.ndarray, sr: int, tonal_fragment_cls) -> int:
    ton = tonal_fragment_cls(
        waveform=y440,
        sr=sr,
        distance="cosine",
        profile="albrecht",
        chromatype="CENS",
    )
    return tonic_label_to_int(str(ton.key))


def extract_mode_float(y: np.ndarray, sr: int, relative_mode_fn) -> float:
    df, _ = relative_mode_fn(
        y=y,
        sr=sr,
        winlen=3,
        hoplen=3,
        cropfirst=0,
        croplast=0,
        distance="cosine",
        profile="albrecht",
        chromatype="CENS",
        remove_percussive=False,
    )
    if df.empty or "tondeltamax" not in df.columns:
        raise RuntimeError("relative_mode did not return scalar tondeltamax")
    return float(df["tondeltamax"].iloc[0])


def extract_features(
    wav_path: Path,
    tonal_fragment_cls,
    relative_mode_fn,
) -> Dict[str, object]:
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    # Same tuning step the project uses before tonal analysis
    y440 = tune_audio_like_relative_mode(y, sr, remove_percussive=False)

    row: Dict[str, object] = {}
    row.update(extract_cqt_chromagram(y440, sr))
    row["key"] = extract_key_int(y440, sr, tonal_fragment_cls)
    row["mode"] = extract_mode_float(y, sr, relative_mode_fn)
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


def finalize_column_order(df: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "movie_id",
        "title",
        "first_genre",
        "cue_index",
        "soundtrack_id",
        "start_time",
        "end_time",
        "embedding_path",
    ] + [f"chromagram_{i}" for i in range(1, 13)] + [
        "key",
        "mode",
    ]

    existing = [c for c in ordered_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def build_movie_features_df(
    movie_id: str,
    title: str,
    first_genre: str,
    group: pd.DataFrame,
    cues_folder: Path,
    tmpdir: Path,
    pbar,
    tonal_fragment_cls,
    relative_mode_fn,
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
                tonal_fragment_cls=tonal_fragment_cls,
                relative_mode_fn=relative_mode_fn,
            )

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

    df = pd.DataFrame(rows)
    return finalize_column_order(df)


def process_all_movies(
    merged: pd.DataFrame,
    cues_folder: Path,
    output_folder: Path,
    tonal_fragment_cls,
    relative_mode_fn,
) -> None:
    processed_movies, errored_movies = load_existing_state(output_folder)
    total_cues = compute_remaining_cues(merged, processed_movies, errored_movies)
    if total_cues <= 0:
        return

    with tempfile.TemporaryDirectory(prefix="tonal_tmp_") as td:
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
                    df = build_movie_features_df(
                        movie_id=movie_id_str,
                        title=title,
                        first_genre=first_genre,
                        group=group,
                        cues_folder=cues_folder,
                        tmpdir=tmpdir,
                        pbar=pbar,
                        tonal_fragment_cls=tonal_fragment_cls,
                        relative_mode_fn=relative_mode_fn,
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


def run():
    args = parse_args()
    require_ffmpeg()

    tonal_fragment_cls, relative_mode_fn, import_path = load_relative_mode_api()
    print(f"Using relative_mode import path: {import_path}")

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
        tonal_fragment_cls=tonal_fragment_cls,
        relative_mode_fn=relative_mode_fn,
    )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)