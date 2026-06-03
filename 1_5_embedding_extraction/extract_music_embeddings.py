#!/usr/bin/env python3
"""
Bulk-extract 2 s-aligned PANN-CNN14 embeddings into a separate output folder,
using the 32 kHz model that AudioTagging expects.

Usage:
    python extract_cnn14.py --root soundtracks \
                            --out_root embeddings \
                            --checkpoint Cnn14_mAP=0.431.pth \
                            --device cuda:0
"""

import argparse, csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from panns_inference import AudioTagging


def get_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root",       required=True,
                    help="Root dir containing soundtrack folders (audio source)")
    ap.add_argument("--out_root",   required=True,
                    help="Root dir to dump .npy embedding files")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to Cnn14_mAP=0.431.pth (32 kHz model)")
    ap.add_argument("--device",     default="cpu",
                    help="Device: e.g. cuda:0 or cpu")
    ap.add_argument("--dtype",      default="float16",
                    choices=["float16","float32"],
                    help="Storage dtype for embeddings")
    ap.add_argument("--win",        type=float, default=2.0,
                    help="Window length in seconds (2 s)")
    ap.add_argument("--hop",        type=float, default=1.0,
                    help="Hop length in seconds (1 s)")
    ap.add_argument("--manifest",   default="manifest.csv",
                    help="Output CSV manifest path")
    return ap.parse_args()


def load_resample(path: Path, target_sr=32000):
    """Load audio, convert to mono, resample to 32 kHz."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:              # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav                       # shape: (samples,)


def chunk_waveform(wav: torch.Tensor, sr: int, win_s: float, hop_s: float):
    """Yield (t_start, chunk) for each window."""
    win_n = int(win_s * sr)
    hop_n = int(hop_s * sr)
    total = wav.shape[0]
    for start in range(0, max(1, total - win_n + 1), hop_n):
        yield start / sr, wav[start:start + win_n]


def pad_to_10s(wav: torch.Tensor, sr=32000):
    """Zero-pad any ≤10 s waveform up to exactly 10 s (320 000 samples)."""
    need = 10 * sr - wav.shape[0]
    if need > 0:
        wav = torch.nn.functional.pad(wav, (0, need))
    return wav


def main():
    args = get_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()

    # Load the PANN-CNN14 32 kHz model via AudioTagging
    at = AudioTagging(checkpoint_path=args.checkpoint,
                      device=args.device)
    at.model.eval()

    manifest = []
    for audio_path in tqdm(list(root.rglob("*.aac")), desc="Tracks"):
        try:
            wav = load_resample(audio_path, target_sr=32000)
            frames, meta = [], []

            # Slide 2 s windows every 1 s
            for t0, chunk in chunk_waveform(wav, 32000, args.win, args.hop):
                chunk10 = pad_to_10s(chunk, sr=32000)
                audio_np = chunk10.cpu().numpy()[None, :]   # shape (1, 320000)
                _, emb = at.inference(audio_np)             # emb.shape == (1,2048)
                emb = emb[0].astype(args.dtype)

                frames.append(emb)
                meta.append({
                    "movie_id":  audio_path.parent.name,
                    "track_rel": str(audio_path.relative_to(root)),
                    "emb_rel":   str(
                        (out_root / audio_path.relative_to(root))
                        .with_suffix(".npy")
                    ),
                    "t_start":   round(t0, 3),
                    "n_dims":    emb.shape[0],
                    "dtype":     args.dtype
                })

            if not frames:
                continue

            # Save per-track embeddings under out_root, preserving folder structure
            rel = audio_path.relative_to(root).with_suffix(".npy")
            npy_path = out_root / rel
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            emb_mat = np.stack(frames, axis=0)  # shape (T, 2048)
            np.save(npy_path, emb_mat)

            manifest.extend(meta)
        except Exception as e:
            print('AUDIO FAILED', audio_path.parent.name, e)

    # Write out the manifest.csv
    keys = ["movie_id", "track_rel", "emb_rel", "t_start", "n_dims", "dtype"]
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Done! Extracted {len(manifest)} frames → {args.manifest}")


if __name__ == "__main__":
    main()