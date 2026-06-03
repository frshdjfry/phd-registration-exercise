import os
import csv
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import cv2
from skimage.feature import local_binary_pattern

VIDEO_DIR = "/Data/Farshad_thesis_films"
OUT_DIR = "basic_features"

SEGMENT_SECONDS = 2
MID_OFFSET = 1.0

FFMPEG_IMAGE_EXT = "jpg"

TARGET_HEIGHT = 240

os.makedirs(OUT_DIR, exist_ok=True)

H_BINS = 18
EDGE_MAG_THRESHOLD_PERCENTILE = 75.0
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"


def append_failed_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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


def probe_video_resolution(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    line = result.stdout.strip().splitlines()[0]
    w_str, h_str = line.strip().split("x")
    return int(w_str), int(h_str)


def detect_crop_rectangle(video_path: str) -> tuple[int, int, int, int] | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-ss", "60",
        "-i", video_path,
        "-t", "120",
        "-an",
        "-sn",
        "-vf", "cropdetect=24:16:0",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stderr + result.stdout

    import re
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", text)
    if not matches:
        return None

    w_str, h_str, x_str, y_str = matches[-1]
    crop_w = int(w_str)
    crop_h = int(h_str)
    crop_x = int(x_str)
    crop_y = int(y_str)
    return crop_x, crop_y, crop_w, crop_h



def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        if "movie_id" in df.columns:
            df = df.set_index("movie_id")
        if df.index.name != "movie_id":
            df.index.name = "movie_id"
        return df
    cols = ["orig_width", "orig_height", "crop_x", "crop_y", "crop_width", "crop_height"]
    df = pd.DataFrame(columns=cols)
    df.index.name = "movie_id"
    return df


def save_metadata(metadata_df: pd.DataFrame, metadata_path: Path):
    df_out = metadata_df.reset_index()
    df_out.to_csv(metadata_path, index=False)


def ensure_metadata_for_movie(
    movie_id: str,
    video_path: str,
    metadata_df: pd.DataFrame,
    metadata_path: Path,
) -> tuple[int, int, int, int]:
    if movie_id in metadata_df.index:
        row = metadata_df.loc[movie_id]
        crop_x = int(row["crop_x"])
        crop_y = int(row["crop_y"])
        crop_w = int(row["crop_width"])
        crop_h = int(row["crop_height"])
        return crop_x, crop_y, crop_w, crop_h

    orig_w, orig_h = probe_video_resolution(video_path)
    crop_rect = detect_crop_rectangle(video_path)
    if crop_rect is None:
        crop_x = 0
        crop_y = 0
        crop_w = orig_w
        crop_h = orig_h
    else:
        crop_x, crop_y, crop_w, crop_h = crop_rect

    metadata_df.loc[movie_id, "orig_width"] = orig_w
    metadata_df.loc[movie_id, "orig_height"] = orig_h
    metadata_df.loc[movie_id, "crop_x"] = crop_x
    metadata_df.loc[movie_id, "crop_y"] = crop_y
    metadata_df.loc[movie_id, "crop_width"] = crop_w
    metadata_df.loc[movie_id, "crop_height"] = crop_h

    save_metadata(metadata_df, metadata_path)

    return crop_x, crop_y, crop_w, crop_h


def ffmpeg_extract_midpoint_frames_gpu(
    video_path: str,
    out_dir: Path,
    crop_rect: tuple[int, int, int, int],
    target_height: int,
    img_ext: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(video_path).resolve()
    src_dir = str(src.parent)
    src_name = src.name
    out_dir_host = str(out_dir.resolve())

    crop_x, crop_y, crop_w, crop_h = crop_rect

    vf = (
        f"fps=1/{SEGMENT_SECONDS},"
        f"crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},"
        f"scale_cuda=w=-2:h={target_height},"
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


def build_frame_context(img: Image.Image) -> dict:
    img_rgb = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "rgb": img_rgb,
        "bgr": img_bgr,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
    }


def luminance_mean(ctx: dict) -> dict:
    lab = ctx["lab"]
    L = lab[:, :, 0]
    return {"luminance_mean": float(L.mean())}


def luminance_std(ctx: dict) -> dict:
    lab = ctx["lab"]
    L = lab[:, :, 0]
    return {"luminance_std": float(L.std())}


def luminance_contrast(ctx: dict) -> dict:
    lab = ctx["lab"]
    L = lab[:, :, 0]
    return {"luminance_contrast": float(L.std())}


def saturation_mean(ctx: dict) -> dict:
    hsv = ctx["hsv"]
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    return {"saturation_mean": float(S.mean())}


def colorfulness(ctx: dict) -> dict:
    lab = ctx["lab"]
    L, a, b = cv2.split(lab)
    a_centered = a - 128.0
    b_centered = b - 128.0
    a_mean = float(a_centered.mean())
    b_mean = float(b_centered.mean())
    a_std = float(a_centered.std())
    b_std = float(b_centered.std())
    sigma = np.sqrt(a_std ** 2 + b_std ** 2)
    mu = np.sqrt(a_mean ** 2 + b_mean ** 2)
    value = float(sigma + 0.3 * mu)
    return {"colorfulness": value}


def warm_cool_ratio(ctx: dict) -> dict:
    hsv = ctx["hsv"]
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    H_deg = H * 2.0
    sat_mask = S > 0.2
    H_sel = H_deg[sat_mask]
    warm_mask = ((H_sel <= 60.0) | (H_sel >= 300.0))
    cool_mask = ((H_sel >= 120.0) & (H_sel <= 240.0))
    warm_count = float(np.count_nonzero(warm_mask))
    cool_count = float(np.count_nonzero(cool_mask))
    ratio = warm_count / (cool_count + 1e-6)
    return {"warm_cool_ratio": ratio}


def hue_histogram(ctx: dict, h_bins: int = H_BINS) -> dict:
    hsv = ctx["hsv"]
    H = hsv[:, :, 0].astype(np.float32)
    hist, _ = np.histogram(H, bins=h_bins, range=(0.0, 180.0), density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    feats = {}
    for i, v in enumerate(hist):
        feats[f"hue_histogram_{i}"] = float(v)
    return feats


def edge_energy(ctx: dict) -> dict:
    gray = ctx["gray"]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return {"edge_energy": float(mag.mean())}


def edge_density(ctx: dict) -> dict:
    gray = ctx["gray"]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = float(np.percentile(mag, EDGE_MAG_THRESHOLD_PERCENTILE))
    if thr <= 0:
        density = 0.0
    else:
        density = float((mag > thr).mean())
    return {"edge_density": density}


def local_binary_pattern_entropy(ctx: dict) -> dict:
    gray = ctx["gray"]
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    eps = 1e-12
    entropy = float(-np.sum(hist * np.log(hist + eps)))
    return {"local_binary_pattern_entropy": entropy}


def local_binary_pattern_histogram(ctx: dict) -> dict:
    gray = ctx["gray"]
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    feats = {}
    for i, v in enumerate(hist):
        feats[f"local_binary_pattern_histogram_{i}"] = float(v)
    return feats


FEATURE_GROUPS = {
    "luminance": [luminance_mean, luminance_std, luminance_contrast],
    "color_basic": [saturation_mean, colorfulness, warm_cool_ratio],
    "color_hist_hsv": [hue_histogram],
    "edges": [edge_energy, edge_density],
    "lbp": [local_binary_pattern_entropy, local_binary_pattern_histogram],
}


def compute_features_for_frame(
    img: Image.Image,
    feature_group_names: list[str],
) -> dict:
    ctx = build_frame_context(img)
    feats: dict[str, float] = {}
    for group_name in feature_group_names:
        funcs = FEATURE_GROUPS[group_name]
        for func in funcs:
            group_feats = func(ctx)
            feats.update(group_feats)
    return feats


def process_one_movie(
    movie_id: str,
    video_path: str,
    frames_root: Path,
    failed_csv: Path,
    feature_group_names: list[str],
    metadata_df: pd.DataFrame,
    metadata_path: Path,
):
    out_path = Path(OUT_DIR) / f"{movie_id}.csv"
    if out_path.exists():
        return

    frame_dir = frames_root / movie_id

    try:
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        crop_rect = ensure_metadata_for_movie(
            movie_id=movie_id,
            video_path=video_path,
            metadata_df=metadata_df,
            metadata_path=metadata_path,
        )

        ffmpeg_extract_midpoint_frames_gpu(
            video_path=video_path,
            out_dir=frame_dir,
            crop_rect=crop_rect,
            target_height=TARGET_HEIGHT,
            img_ext=FFMPEG_IMAGE_EXT,
        )

        frame_files = sorted(frame_dir.glob(f"frame_*.{FFMPEG_IMAGE_EXT}"))
        if not frame_files:
            raise RuntimeError("No frames extracted by ffmpeg")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(out_path, "w", newline="", encoding="utf-8")
        writer = None

        try:
            for i, fp in enumerate(frame_files):
                t = MID_OFFSET + i * SEGMENT_SECONDS
                img = Image.open(fp).convert("RGB")

                feats = compute_features_for_frame(img, feature_group_names)
                row = {
                    "movie_id": movie_id,
                    "timestamp": float(t),
                }
                row.update(feats)

                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(row)
        finally:
            if writer is not None:
                csv_file.close()

    except Exception as e:
        append_failed_csv(
            failed_csv,
            {
                "movie_id": movie_id,
                "video_path": video_path,
                "error": str(e),
            }
        )
        if out_path.exists():
            out_path.unlink(missing_ok=True)
    finally:
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", required=True, help="Genre to process (e.g. Drama)")
    ap.add_argument("--parquet", default="movies.parquet", help="Path to movies.parquet")
    ap.add_argument("--frames_root", default="/dev/shm/frames_by_genre", help="Root dir for extracted frames")
    ap.add_argument(
        "--features",
        default="luminance,color_basic,color_hist_hsv,edges,lbp",
        help=(
            "Comma-separated list of feature groups to extract. "
            f"Available: {','.join(FEATURE_GROUPS.keys())}"
        ),
    )
    ap.add_argument(
        "--metadata",
        default="video_metadata.csv",
        help="Path to video metadata parquet (will be created/updated)",
    )
    args = ap.parse_args()

    feature_group_names = [f.strip() for f in args.features.split(",") if f.strip()]
    for name in feature_group_names:
        if name not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group '{name}'. "
                f"Available: {', '.join(FEATURE_GROUPS.keys())}"
            )

    metadata_path = Path(args.metadata)
    metadata_df = load_metadata(metadata_path)

    genre_slug = args.genre.strip().lower().replace(" ", "_")
    video_dir = Path(VIDEO_DIR)

    failed_csv = Path(f"failed_movies_{genre_slug}_basic_features.csv")
    frames_root = Path(args.frames_root) / genre_slug

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

    for mid, vp in tqdm(jobs, desc=f"Genre={args.genre} (basic features)"):
        process_one_movie(
            movie_id=mid,
            video_path=vp,
            frames_root=frames_root,
            failed_csv=failed_csv,
            feature_group_names=feature_group_names,
            metadata_df=metadata_df,
            metadata_path=metadata_path,
        )

    save_metadata(metadata_df, metadata_path)


if __name__ == "__main__":
    main()
