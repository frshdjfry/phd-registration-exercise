import os
import csv
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import math


VIDEO_DIR = "/Data/Farshad_thesis_films"


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


def load_scenes_from_csv(scenes_csv_path, movie_id):
    scenes = []
    with Path(scenes_csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("movie_id") != str(movie_id):
                continue
            start_time = float(row["scene_start_time"])
            end_time = float(row["scene_end_time"])
            scenes.append(
                {
                    "scene_index": int(row["scene_index"]),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
    if not scenes:
        raise RuntimeError(f"No scenes found for movie_id={movie_id} in {scenes_csv_path}")
    scenes_sorted = sorted(scenes, key=lambda s: s["scene_index"])
    video_duration = max(s["end_time"] for s in scenes_sorted)
    return scenes_sorted, video_duration


def probe_fps(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rate_str = result.stdout.strip()
    if "/" in rate_str:
        num, den = rate_str.split("/")
        return float(num) / float(den)
    return float(rate_str)


def build_intervals_from_scenes(
    scenes,
    target_dt,
    min_dt,
    fps,
    boundary_margin_frames: float = 1.0,
):
    intervals = []
    frame_dt = 1.0 / fps
    margin = boundary_margin_frames * frame_dt

    for s in scenes:
        raw_start = s["start_time"]
        raw_end = s["end_time"]

        shot_start = raw_start + margin
        shot_end = raw_end - margin

        if shot_end <= shot_start + min_dt:
            continue

        t0 = shot_start
        while True:
            t1 = t0 + target_dt
            if t1 > shot_end:
                t1 = shot_end

            dt = t1 - t0
            if dt < min_dt:
                break

            mid_time = 0.5 * (t0 + t1)
            intervals.append(
                {
                    "t0": t0,
                    "t1": t1,
                    "delta_t": dt,
                    "mid_time": mid_time,
                }
            )

            if t1 >= shot_end:
                break

            t0 = t1

    return intervals


def ffmpeg_extract_frames_gpu(
    video_path: str,
    times: list[float],
    fps: float,
    out_dir: Path,
    movie_id: str,
    flow_resize_width: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(video_path).resolve()
    src_dir = str(src.parent)
    src_name = src.name
    out_dir_host = str(out_dir.resolve())

    # times must be sorted
    times = sorted(times)
    frame_indices = [int(round(t * fps)) for t in times]

    # build select expression: eq(n\,idx1)+eq(n\,idx2)+...
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)

    script_path = out_dir / "ff_filters.txt"
    with script_path.open("w", encoding="utf-8") as f:
        if flow_resize_width is not None and flow_resize_width > 0:
            f.write(
                "[0:v]"
                f"select='{select_expr}',"
                f"scale_cuda=w={int(flow_resize_width)}:h=-2,"
                "hwdownload,format=nv12,format=rgb24"
                "[outv]\n"
            )
        else:
            f.write(
                f"[0:v]select='{select_expr}'[outv]\n"
            )

    script_name = script_path.name

    cmd = [
        "docker",
        "run",
        "--gpus",
        "all",
        "--rm",
        "-v",
        f"{src_dir}:/data/input:ro",
        "-v",
        f"{out_dir_host}:/data/output",
        "jrottenberg/ffmpeg:8-nvidia",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        f"/data/input/{src_name}",
        "-filter_complex_script",
        f"/data/output/{script_name}",
        "-map",
        "[outv]",
        "-vsync",
        "0",
        "-start_number",
        "0",
        "-q:v",
        "2",
        "/data/output/frame_%06d.jpg",
    ]

    subprocess.run(cmd, check=True)

    time_to_path: dict[float, str] = {}
    for idx, t in enumerate(times):
        out_path = out_dir / f"frame_{idx:06d}.jpg"
        time_to_path[t] = str(out_path)
    return time_to_path

def compute_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return flow[..., 0], flow[..., 1]


def decompose_affine(M):
    a11, a12, tx = M[0]
    a21, a22, ty = M[1]
    sx = math.sqrt(a11 * a11 + a21 * a21)
    sy = math.sqrt(a12 * a12 + a22 * a22)
    if sx + sy == 0:
        scale = 1.0
    else:
        scale = (sx + sy) / 2.0
    rot = math.degrees(math.atan2(a21, a11))
    return tx, ty, scale, rot


def prepare_grayscale_pair(frame0, frame1):
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray0.shape[:2]
    diag = math.sqrt(w0 * w0 + h0 * h0)
    return gray0, gray1, diag


def compute_flow_features(gray0, gray1, num_bins, mag_thresh):
    flow_x, flow_y = compute_optical_flow(gray0, gray1)
    mag, ang = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=False)
    motion_mask = mag >= mag_thresh
    if np.count_nonzero(motion_mask) == 0:
        return None
    mags = mag[motion_mask]
    angles = ang[motion_mask]
    mean_mag = float(mags.mean())
    p95_mag = float(np.percentile(mags, 95))
    h, w = gray0.shape[:2]
    motion_pixel_ratio = float(mags.size) / float(w * h)
    hist, _ = np.histogram(angles - math.pi, bins=num_bins, range=(-math.pi, math.pi))
    if hist.sum() > 0:
        hist = hist.astype(np.float32) / float(hist.sum())
    else:
        hist = np.zeros(num_bins, dtype=np.float32)
    return {
        "mean_mag": mean_mag,
        "p95_mag": p95_mag,
        "motion_pixel_ratio": motion_pixel_ratio,
        "hist": hist,
    }


def compute_camera_features(gray0, gray1):
    p0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if p0 is None or len(p0) < 6:
        return 0.0, 0.0, 1.0

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None)
    if p1 is None or st is None:
        return 0.0, 0.0, 1.0

    st = st.reshape(-1)
    good_old = p0[st == 1].reshape(-1, 2)
    good_new = p1[st == 1].reshape(-1, 2)
    if good_old.shape[0] < 6:
        return 0.0, 0.0, 1.0

    src = good_old.reshape(-1, 1, 2).astype(np.float32)
    dst = good_new.reshape(-1, 1, 2).astype(np.float32)
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if M is None:
        return 0.0, 0.0, 1.0

    tx, ty, scale, rot = decompose_affine(M)
    cam_tx = -tx
    cam_ty = -ty
    cam_scale = scale
    return cam_tx, cam_ty, cam_scale


def compute_motion_sample(frame0, frame1, num_bins, mag_thresh):
    gray0, gray1, diag = prepare_grayscale_pair(frame0, frame1)
    flow_feats = compute_flow_features(gray0, gray1, num_bins, mag_thresh)
    if flow_feats is None:
        return None
    cam_tx, cam_ty, cam_scale = compute_camera_features(gray0, gray1)
    return {
        "mean_mag": flow_feats["mean_mag"],
        "p95_mag": flow_feats["p95_mag"],
        "diag": diag,
        "motion_pixel_ratio": flow_feats["motion_pixel_ratio"],
        "hist": flow_feats["hist"],
        "cam_tx": cam_tx,
        "cam_ty": cam_ty,
        "cam_scale": cam_scale,
    }


def extract_motion_samples_from_frames(time_to_path, intervals, num_bins, mag_thresh):
    samples = []
    for iv in intervals:
        t0 = iv["t0"]
        t1 = iv["t1"]
        dt = iv["delta_t"]
        mid_time = iv["mid_time"]
        p0 = time_to_path.get(t0)
        p1 = time_to_path.get(t1)
        if p0 is None or p1 is None:
            continue
        frame0 = cv2.imread(p0)
        frame1 = cv2.imread(p1)
        if frame0 is None or frame1 is None:
            continue
        ms = compute_motion_sample(frame0, frame1, num_bins, mag_thresh)
        if ms is None:
            continue
        flow_mean_rate = (ms["mean_mag"] / ms["diag"]) / dt
        flow_p95_rate = (ms["p95_mag"] / ms["diag"]) / dt
        camera_translation_x_rate = (ms["cam_tx"] / ms["diag"]) / dt
        camera_translation_y_rate = (ms["cam_ty"] / ms["diag"]) / dt
        camera_scale_change_rate = (ms["cam_scale"] - 1.0) / dt
        samples.append(
            {
                "time": mid_time,
                "delta_t": dt,
                "flow_mean_rate": flow_mean_rate,
                "flow_p95_rate": flow_p95_rate,
                "motion_pixel_ratio": ms["motion_pixel_ratio"],
                "camera_translation_x_rate": camera_translation_x_rate,
                "camera_translation_y_rate": camera_translation_y_rate,
                "camera_scale_change_rate": camera_scale_change_rate,
                "hist": ms["hist"],
            }
        )
    return samples


def aggregate_segments(samples, segment_duration, video_duration, num_bins):
    num_segments = int(math.ceil(video_duration / segment_duration))
    segments = []
    for idx in range(num_segments):
        seg_start = idx * segment_duration
        seg_end = seg_start + segment_duration
        seg_samples = [s for s in samples if seg_start <= s["time"] < seg_end]
        if not seg_samples:
            hist = np.zeros(num_bins, dtype=np.float32)
            segments.append(
                {
                    "index": idx,
                    "start": seg_start,
                    "end": seg_end,
                    "flow_mean_rate": 0.0,
                    "flow_p95_rate": 0.0,
                    "motion_pixel_ratio": 0.0,
                    "camera_translation_x_rate": 0.0,
                    "camera_translation_y_rate": 0.0,
                    "camera_scale_change_rate": 0.0,
                    "hist": hist,
                }
            )
            continue
        total_dt = sum(s["delta_t"] for s in seg_samples)
        if total_dt <= 0:
            hist = np.zeros(num_bins, dtype=np.float32)
            segments.append(
                {
                    "index": idx,
                    "start": seg_start,
                    "end": seg_end,
                    "flow_mean_rate": 0.0,
                    "flow_p95_rate": 0.0,
                    "motion_pixel_ratio": 0.0,
                    "camera_translation_x_rate": 0.0,
                    "camera_translation_y_rate": 0.0,
                    "camera_scale_change_rate": 0.0,
                    "hist": hist,
                }
            )
            continue
        flow_mean_rate = sum(s["flow_mean_rate"] * s["delta_t"] for s in seg_samples) / total_dt
        flow_p95_rate = sum(s["flow_p95_rate"] * s["delta_t"] for s in seg_samples) / total_dt
        motion_pixel_ratio = sum(s["motion_pixel_ratio"] * s["delta_t"] for s in seg_samples) / total_dt
        camera_translation_x_rate = sum(s["camera_translation_x_rate"] * s["delta_t"] for s in seg_samples) / total_dt
        camera_translation_y_rate = sum(s["camera_translation_y_rate"] * s["delta_t"] for s in seg_samples) / total_dt
        camera_scale_change_rate = sum(s["camera_scale_change_rate"] * s["delta_t"] for s in seg_samples) / total_dt
        hist_acc = np.zeros(num_bins, dtype=np.float32)
        for s in seg_samples:
            hist_acc += s["hist"] * s["delta_t"]
        if hist_acc.sum() > 0:
            hist = hist_acc / float(hist_acc.sum())
        else:
            hist = np.zeros(num_bins, dtype=np.float32)
        segments.append(
            {
                "index": idx,
                "start": seg_start,
                "end": seg_end,
                "flow_mean_rate": round(float(flow_mean_rate), 3),
                "flow_p95_rate": round(float(flow_p95_rate), 3),
                "motion_pixel_ratio": round(float(motion_pixel_ratio), 3),
                "camera_translation_x_rate": round(float(camera_translation_x_rate), 3),
                "camera_translation_y_rate": round(float(camera_translation_y_rate), 3),
                "camera_scale_change_rate": round(float(camera_scale_change_rate), 3),
                "hist": hist,
            }
        )
    return segments


def write_segments_csv(segments, num_bins, output_csv_path):
    fieldnames = [
        "segment_index",
        "segment_start_time_seconds",
        "segment_end_time_seconds",
        "flow_magnitude_mean_fraction_of_diagonal_per_second",
        "flow_magnitude_p95_fraction_of_diagonal_per_second",
        "motion_pixel_ratio",
        "camera_translation_x_fraction_of_diagonal_per_second",
        "camera_translation_y_fraction_of_diagonal_per_second",
        "camera_scale_change_per_second",
    ]
    hist_names = [f"flow_direction_histogram_bin_{i}" for i in range(num_bins)]
    fieldnames.extend(hist_names)
    p = Path(output_csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments:
            row = {
                "segment_index": seg["index"],
                "segment_start_time_seconds": seg["start"],
                "segment_end_time_seconds": seg["end"],
                "flow_magnitude_mean_fraction_of_diagonal_per_second": seg["flow_mean_rate"],
                "flow_magnitude_p95_fraction_of_diagonal_per_second": seg["flow_p95_rate"],
                "motion_pixel_ratio": seg["motion_pixel_ratio"],
                "camera_translation_x_fraction_of_diagonal_per_second": seg["camera_translation_x_rate"],
                "camera_translation_y_fraction_of_diagonal_per_second": seg["camera_translation_y_rate"],
                "camera_scale_change_per_second": seg["camera_scale_change_rate"],
            }
            for i, v in enumerate(seg["hist"]):
                row[f"flow_direction_histogram_bin_{i}"] = float(v)
            writer.writerow(row)


def process_one_movie(
    movie_id: str,
    video_path: str,
    scenes_csv_path: str,
    frames_root: Path,
    failed_csv: Path,
    output_root: Path,
    segment_duration_seconds: float,
    flow_resize_width: int,
    num_direction_bins: int,
    motion_magnitude_threshold: float,
    target_motion_dt: float,
    min_dt: float,
):
    out_path = output_root / f"{movie_id}.csv"
    if out_path.exists():
        return

    frame_dir = frames_root / movie_id

    try:
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        fps = probe_fps(video_path)
        scenes, video_duration = load_scenes_from_csv(scenes_csv_path, movie_id)
        intervals = build_intervals_from_scenes(
            scenes=scenes,
            target_dt=target_motion_dt,
            min_dt=min_dt,
            fps=fps,
            boundary_margin_frames=1.0,
        )
        if not intervals:
            raise RuntimeError("No motion intervals built from scenes")

        all_times = []
        for iv in intervals:
            all_times.append(iv["t0"])
            all_times.append(iv["t1"])
        unique_times = sorted(set(all_times))

        time_to_path = ffmpeg_extract_frames_gpu(
            video_path=video_path,
            times=unique_times,
            fps=fps,
            out_dir=frame_dir,
            movie_id=movie_id,
            flow_resize_width=flow_resize_width,
        )

        samples = extract_motion_samples_from_frames(
            time_to_path=time_to_path,
            intervals=intervals,
            num_bins=num_direction_bins,
            mag_thresh=motion_magnitude_threshold,
        )

        segments = aggregate_segments(
            samples,
            segment_duration=segment_duration_seconds,
            video_duration=video_duration,
            num_bins=num_direction_bins,
        )
        write_segments_csv(segments, num_direction_bins, out_path)

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
    ap.add_argument("--frames_root", default="/dev/shm/motion_frames_by_genre", help="Root dir for extracted frames")
    ap.add_argument("--scenes_csv_path", default="scenes.csv", help="Path to scenes.csv")
    ap.add_argument("--output_root", default="basic_motion_features", help="Output dir for motion CSVs")
    ap.add_argument("--segment_duration_seconds", type=float, default=2.0)
    ap.add_argument("--flow_resize_width", type=int, default=360)
    ap.add_argument("--num_direction_bins", type=int, default=8)
    ap.add_argument("--motion_magnitude_threshold", type=float, default=0.5)
    ap.add_argument("--target_motion_dt", type=float, default=0.5)
    ap.add_argument("--min_dt", type=float, default=0.1)
    args = ap.parse_args()

    video_dir = Path(VIDEO_DIR)
    output_root = Path(args.output_root)
    genre_slug = args.genre.strip().lower().replace(" ", "_")
    frames_root = Path(args.frames_root) / genre_slug
    failed_csv = Path(f"failed_movies_{genre_slug}_motion_features.csv")

    movie_ids = iter_movies_for_genre(args.parquet, args.genre)

    jobs = []
    for mid in movie_ids:
        vp = find_movie_file(mid, video_dir)
        if vp is not None:
            jobs.append((mid, vp))
        else:
            append_failed_csv(
                failed_csv,
                {"movie_id": mid, "video_path": "", "error": "video file not found in VIDEO_DIR"},
            )

    for mid, vp in tqdm(jobs, desc=f"Genre={args.genre} (motion features)"):
        process_one_movie(
            movie_id=mid,
            video_path=vp,
            scenes_csv_path=args.scenes_csv_path,
            frames_root=frames_root,
            failed_csv=failed_csv,
            output_root=output_root,
            segment_duration_seconds=args.segment_duration_seconds,
            flow_resize_width=args.flow_resize_width,
            num_direction_bins=args.num_direction_bins,
            motion_magnitude_threshold=args.motion_magnitude_threshold,
            target_motion_dt=args.target_motion_dt,
            min_dt=args.min_dt,
        )


if __name__ == "__main__":
    main()