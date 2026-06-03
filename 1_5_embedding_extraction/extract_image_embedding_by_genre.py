import os
import json
import csv
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import pandas as pd

# =========================
# Configuration
# =========================

VIDEO_DIR = "/Data/Farshad_thesis_films"
OUT_DIR = "image_embeddings"

SEGMENT_SECONDS = 2
MID_OFFSET = 1.0  # we want frames at 1,3,5,... seconds

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
IMAGE_SIZE = 224  # expected by ViT-L-14

FFMPEG_IMAGE_EXT = "jpg"   # jpg is faster/smaller; png is lossless but heavier

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Model
# =========================

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE
)
model.eval()
normalize = preprocess.transforms[-1]  # only normalization

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


def letterbox_pad(img: Image.Image, size: int = 224) -> Image.Image:
    """Resize preserving aspect ratio, pad with black to (size, size)."""
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


def find_movie_file(movie_id: str, video_dir: Path) -> str | None:
    for ext in (".mp4", ".mkv", ".MP4", ".MKV"):
        p = video_dir / f"{movie_id}{ext}"
        if p.exists():
            return str(p)
    return None


def iter_movies_for_genre(parquet_path: str, genre: str):
    df = pd.read_parquet(parquet_path, columns=["movie_id", "first_genre"])
    g = genre.strip().lower()
    mask = df["first_genre"].fillna("").str.lower().eq(g)
    return df.loc[mask, "movie_id"].tolist()

def ffmpeg_extract_midpoint_frames_gpu(
    video_path: str,
    out_dir: Path,
    width: int = 224,
    img_ext: str = "png",
):
    """
    Extract frames at 1,3,5,... seconds using:
    GPU decode + GPU scale + GPU pad (pad_cuda),
    then download once to CPU for image output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(video_path).resolve()
    src_dir = str(src.parent)
    src_name = src.name
    out_dir_host = str(out_dir.resolve())

    vf = (
        f"fps=1/{SEGMENT_SECONDS},"
        f"scale_cuda=w={width}:h=-2,"
        f"pad_cuda={width}:{width}:(ow-iw)/2:(oh-ih)/2,"
        f"hwdownload,format=nv12,format=rgb24"
    )
    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{src_dir}:/data/input:ro",
        "-v", f"{out_dir_host}:/data/output",
        "jrottenberg/ffmpeg:8-nvidia",
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(MID_OFFSET),
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", f"/data/input/{src_name}",
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-c:v", "png",
        f"/data/output/frame_%06d.{img_ext}",
    ]

    subprocess.run(cmd, check=True)



def process_one_movie(movie_id: str, video_path: str, frames_root: Path, failed_csv: Path):
    out_path = Path(OUT_DIR) / f"{movie_id}.npz"
    if out_path.exists():
        return  # resume-safe

    # per-movie frame dir to avoid collisions
    frame_dir = frames_root / movie_id

    try:
        # Clean stale frames if any
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        ffmpeg_extract_midpoint_frames_gpu(
            video_path=video_path,
            out_dir=frame_dir,
            width=IMAGE_SIZE,
            img_ext=FFMPEG_IMAGE_EXT,
        )

        frame_files = sorted(frame_dir.glob(f"frame_*.{FFMPEG_IMAGE_EXT}"))
        if not frame_files:
            raise RuntimeError("No frames extracted by ffmpeg")

        embeddings = []
        saved_timestamps = []
        batch_imgs = []

        # timestamps are deterministic: 1,3,5,... aligned with extracted frames order
        # frame_000001 corresponds to t=1, frame_000002 -> t=3, etc.
        for i, fp in enumerate(frame_files):
            t = MID_OFFSET + i * SEGMENT_SECONDS
            img = Image.open(fp).convert("RGB")  # already 224x224 padded by ffmpeg
            # If you trust ffmpeg pad, you can skip PIL letterbox
            # img = letterbox_pad(img, IMAGE_SIZE)

            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_t = normalize(img_t)
            batch_imgs.append(img_t)
            saved_timestamps.append(float(t))

            if len(batch_imgs) == BATCH_SIZE:
                batch = torch.stack(batch_imgs).to(DEVICE)
                with torch.no_grad():
                    feats = model.encode_image(batch)
                embeddings.append(feats.cpu().numpy())
                batch_imgs.clear()

        if batch_imgs:
            batch = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_image(batch)
            embeddings.append(feats.cpu().numpy())

        if not embeddings:
            raise RuntimeError("No embeddings produced")

        emb = np.concatenate(embeddings, axis=0)
        np.savez(out_path, embeddings=emb, timestamps=np.array(saved_timestamps, dtype=np.float32))

    except Exception as e:
        append_failed_csv(
            failed_csv,
            {
                "movie_id": movie_id,
                "video_path": video_path,
                "error": str(e),
            }
        )
    finally:
        # always clean frames to avoid filling /dev/shm
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", required=True, help="Genre to process (e.g. Drama)")
    ap.add_argument("--parquet", default="movies.parquet", help="Path to movies.parquet")
    ap.add_argument("--frames_root", default="/dev/shm/frames_by_genre", help="Root dir for extracted frames")
    args = ap.parse_args()

    genre_slug = args.genre.strip().lower().replace(" ", "_")
    video_dir = Path(VIDEO_DIR)

    failed_csv = Path(f"failed_movies_{genre_slug}.csv")
    frames_root = Path(args.frames_root) / genre_slug  # per-genre isolation (no collisions)

    movie_ids = iter_movies_for_genre(args.parquet, args.genre)

    jobs = []
    for mid in movie_ids:
        vp = find_movie_file(mid, video_dir)
        if vp is not None:
            jobs.append((mid, vp))
        else:
            append_failed_csv(
                failed_csv,
                {"movie_id": mid, "video_path": "", "error": "video file not found in VIDEO_DIR"}
            )

    for mid, vp in tqdm(jobs, desc=f"Genre={args.genre}"):
        process_one_movie(mid, vp, frames_root=frames_root, failed_csv=failed_csv)


if __name__ == "__main__":
    main()
