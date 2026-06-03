#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# =========================
# Configuration
# =========================

OUT_DIR = "dialogue_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 256


# =========================
# Utilities
# =========================

def append_failed_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def iter_movies_for_genre(parquet_path: str, genre: str):
    df = pd.read_parquet(parquet_path, columns=["movie_id", "first_genre"])
    g = genre.strip().lower()
    mask = df["first_genre"].fillna("").str.lower().eq(g)
    return df.loc[mask, "movie_id"].astype(str).tolist()


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B,1]
    return summed / counts


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**tok)
        sent = mean_pooling(out.last_hidden_state, tok["attention_mask"])

        if normalize:
            sent = torch.nn.functional.normalize(sent, p=2, dim=1)

        embs.append(sent.detach().cpu().to(torch.float32).numpy())

    # If no texts, return empty
    if not embs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    return np.vstack(embs)


# =========================
# Per-movie processing
# =========================

def process_one_movie(
    movie_id: str,
    dialogues_path: str,
    text_col: str,
    start_col: str,
    end_col: str,
    movie_col: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    failed_csv: Path,
    batch_size: int,
    max_length: int,
    normalize: bool,
    min_chars: int = 1,
):
    out_path = Path(OUT_DIR) / f"{movie_id}.npz"
    if out_path.exists():
        return  # resume-safe

    try:
        # Read only needed columns for this movie (fast)
        # NOTE: Parquet row-group filtering isn't guaranteed unless partitioned,
        # but selecting columns still helps a lot.
        df = pd.read_parquet(dialogues_path, columns=[movie_col, text_col, start_col, end_col])

        # Filter movie
        m = df[movie_col].astype(str).eq(str(movie_id))
        dfm = df.loc[m, [text_col, start_col, end_col]].copy()
        if dfm.empty:
            raise RuntimeError("No dialogue rows for this movie_id")

        # Keep original row ids so the viewer can map back to parquet index
        # (If you don't need this, you can remove it.)
        dfm["row_id"] = dfm.index.astype(np.int64)

        # Basic text filtering
        dfm[text_col] = dfm[text_col].astype(str)
        dfm = dfm[dfm[text_col].str.len() >= min_chars].copy()
        if dfm.empty:
            raise RuntimeError("All rows empty after text length filter")

        texts = dfm[text_col].tolist()
        start = dfm[start_col].astype(np.float32).to_numpy()
        end = dfm[end_col].astype(np.float32).to_numpy()
        row_id = dfm["row_id"].to_numpy(np.int64)

        emb = encode_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=DEVICE,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
        ).astype(np.float32)

        if emb.shape[0] != len(start):
            raise RuntimeError(f"Embeddings rows ({emb.shape[0]}) != dialogues rows ({len(start)})")

        np.savez_compressed(
            out_path,
            embeddings=emb,
            row_id=row_id,
            start_time=start,
            end_time=end,
            movie_id=np.array([str(movie_id)]),
        )

    except Exception as e:
        append_failed_csv(
            failed_csv,
            {"movie_id": movie_id, "error": str(e)}
        )


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", required=True, help="Genre to process (e.g. Drama)")
    ap.add_argument("--movies_parquet", default="movies.parquet", help="Path to movies.parquet")
    ap.add_argument("--dialogues_parquet", default="dialogues_clean.parquet", help="Path to dialogues parquet")
    ap.add_argument("--movie-col", default="movie_id", help="Movie id column name in dialogues parquet")
    ap.add_argument("--text-col", default="text_clean", help="Text column to embed")
    ap.add_argument("--start-col", default="start_time", help="Start time column name")
    ap.add_argument("--end-col", default="end_time", help="End time column name")

    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N movies (debug)")

    args = ap.parse_args()

    genre_slug = args.genre.strip().lower().replace(" ", "_")
    failed_csv = Path(f"failed_dialogues_{genre_slug}.csv")

    movie_ids = iter_movies_for_genre(args.movies_parquet, args.genre)
    if args.limit and len(movie_ids) > args.limit:
        movie_ids = movie_ids[: args.limit]

    # Load model once (like your image script)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(DEVICE).eval()

    for mid in tqdm(movie_ids, desc=f"Genre={args.genre} (dialogue embeddings)"):
        process_one_movie(
            movie_id=str(mid),
            dialogues_path=args.dialogues_parquet,
            text_col=args.text_col,
            start_col=args.start_col,
            end_col=args.end_col,
            movie_col=args.movie_col,
            tokenizer=tokenizer,
            model=model,
            failed_csv=failed_csv,
            batch_size=args.batch_size,
            max_length=args.max_length,
            normalize=args.normalize,
        )


if __name__ == "__main__":
    main()


# python extract_dialogue_embeddings_by_genre.py --genre Drama --movies_parquet movies.parquet --dialogues_parquet dialogues_clean.parquet  --normalize
