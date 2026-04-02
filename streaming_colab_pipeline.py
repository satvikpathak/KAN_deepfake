#!/usr/bin/env python3
"""Colab-safe full-dataset training pipeline for phase-based deepfake detection.

Why this exists:
- The stage notebooks currently materialize full arrays in RAM (X_l -> np.array -> PCA.fit_transform),
  which can crash Colab on large datasets.

What this script does:
1) Extracts phase maps into fixed-size on-disk shards (float16 .npy files).
2) Builds a manifest with train/val/test split indices.
3) Trains a CNN directly from memory-mapped shards for 100 epochs (default), with AMP,
   gradient accumulation, and checkpointing.

This keeps RAM usage bounded and supports full-dataset training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LUMA_R, LUMA_G, LUMA_B = 0.2989, 0.5870, 0.1140
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size_from_train: float = 0.15
    random_state: int = 42


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    grad_accum_steps: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_label(path: str, generator: str = "") -> int:
    low = f"{generator} {path}".lower()
    return 0 if "real" in low else 1


def discover_metadata(input_dir: str) -> pd.DataFrame:
    """Return dataframe with columns image_path,target,generator.

    Tries metadata CSVs first; falls back to directory scan if no metadata exists.
    """
    meta_files = glob.glob(os.path.join(input_dir, "**", "metadata.csv"), recursive=True)
    rows: List[Dict[str, object]] = []

    for mf in meta_files:
        try:
            df = pd.read_csv(mf)
        except Exception:
            continue

        base = os.path.dirname(mf)
        generator = os.path.basename(base)

        # Candidate path columns in likely priority order
        path_col = None
        for c in ("image_path", "path", "filepath", "file_path", "filename", "image", "file"):
            if c in df.columns:
                path_col = c
                break

        if path_col is None:
            continue

        target_col = "target" if "target" in df.columns else None

        for _, r in df.iterrows():
            raw_p = str(r[path_col])
            p = raw_p if os.path.isabs(raw_p) else os.path.join(base, raw_p)
            if not os.path.exists(p):
                # Try one level up when metadata uses nested relative paths
                p2 = os.path.join(os.path.dirname(base), raw_p)
                if os.path.exists(p2):
                    p = p2
            if not os.path.exists(p):
                continue

            if target_col is not None:
                try:
                    y = int(r[target_col])
                except Exception:
                    y = infer_label(p, generator)
            else:
                y = infer_label(p, generator)

            rows.append({"image_path": p, "target": y, "generator": generator})

    if rows:
        mdf = pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
        return mdf

    # Fallback: walk all image files and infer labels from path text.
    for root, _, files in os.walk(input_dir):
        generator = os.path.basename(root)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMG_EXTS:
                continue
            p = os.path.join(root, f)
            rows.append({"image_path": p, "target": infer_label(p, generator), "generator": generator})

    if not rows:
        raise FileNotFoundError(f"No metadata.csv or images found under: {input_dir}")

    return pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def extract_phase_map(path: str, image_size: int = 256) -> Optional[np.ndarray]:
    try:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        gray = LUMA_R * rgb[:, :, 0] + LUMA_G * rgb[:, :, 1] + LUMA_B * rgb[:, :, 2]
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        phase = np.angle(fft)
        mn, mx = phase.min(), phase.max()
        if mx <= mn:
            return np.zeros((image_size, image_size), dtype=np.float32)
        return ((phase - mn) / (mx - mn)).astype(np.float32)
    except Exception:
        return None


def _safe_stratify(y: Sequence[int]) -> Optional[np.ndarray]:
    u = np.unique(np.asarray(y))
    return np.asarray(y) if len(u) > 1 else None


def build_split_column(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    idx = np.arange(len(df))
    y = df["target"].to_numpy()

    trv_idx, te_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=_safe_stratify(y),
    )

    y_trv = y[trv_idx]
    tr_idx, va_idx = train_test_split(
        trv_idx,
        test_size=cfg.val_size_from_train,
        random_state=cfg.random_state,
        stratify=_safe_stratify(y_trv),
    )

    split = np.array(["train"] * len(df), dtype=object)
    split[va_idx] = "val"
    split[te_idx] = "test"

    out = df.copy()
    out["split"] = split
    return out


def extract_to_shards(
    input_dir: str,
    cache_dir: str,
    shard_size: int = 2048,
    image_size: int = 256,
    out_dtype: str = "float16",
    split_cfg: SplitConfig = SplitConfig(),
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    shard_dir = os.path.join(cache_dir, "phase_shards")
    os.makedirs(shard_dir, exist_ok=True)

    mdf = discover_metadata(input_dir)
    mdf = build_split_column(mdf, split_cfg)

    dtype = np.float16 if out_dtype == "float16" else np.float32
    x_buf: List[np.ndarray] = []
    y_buf: List[int] = []
    path_buf: List[str] = []
    split_buf: List[str] = []
    gen_buf: List[str] = []

    shard_id = 0
    row_count = 0
    failed = 0

    manifest_path = os.path.join(shard_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["shard_id", "offset", "target", "split", "image_path", "generator"])

        def flush() -> None:
            nonlocal shard_id, row_count
            if not x_buf:
                return
            x = np.stack(x_buf, axis=0).astype(dtype, copy=False)
            y = np.asarray(y_buf, dtype=np.uint8)
            np.save(os.path.join(shard_dir, f"x_{shard_id:05d}.npy"), x)
            np.save(os.path.join(shard_dir, f"y_{shard_id:05d}.npy"), y)

            for off, (t, sp, p, g) in enumerate(zip(y_buf, split_buf, path_buf, gen_buf)):
                writer.writerow([shard_id, off, int(t), sp, p, g])
                row_count += 1

            shard_id += 1
            x_buf.clear()
            y_buf.clear()
            path_buf.clear()
            split_buf.clear()
            gen_buf.clear()

        for _, r in mdf.iterrows():
            phase = extract_phase_map(str(r["image_path"]), image_size=image_size)
            if phase is None:
                failed += 1
                continue
            x_buf.append(phase)
            y_buf.append(int(r["target"]))
            path_buf.append(str(r["image_path"]))
            split_buf.append(str(r["split"]))
            gen_buf.append(str(r["generator"]))

            if len(x_buf) >= shard_size:
                flush()

        flush()

    stats = {
        "samples_written": row_count,
        "failed_images": failed,
        "shards": shard_id,
        "image_size": image_size,
        "dtype": out_dtype,
        "split_counts": pd.read_csv(manifest_path)["split"].value_counts().to_dict() if row_count > 0 else {},
    }

    with open(os.path.join(shard_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if row_count == 0:
        raise RuntimeError("No valid samples were extracted. Check dataset paths/metadata.")

    print(f"[extract] wrote {row_count} samples to {shard_id} shards | failed={failed}")
    return shard_dir


class PhaseShardDataset(Dataset):
    def __init__(self, shard_dir: str, split: str):
        self.shard_dir = shard_dir
        mf = pd.read_csv(os.path.join(shard_dir, "manifest.csv"))
        self.df = mf[mf["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in manifest.")

        self._x_cache: Dict[int, np.ndarray] = {}
        self._y_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_shard(self, shard_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if shard_id not in self._x_cache:
            x = np.load(os.path.join(self.shard_dir, f"x_{shard_id:05d}.npy"), mmap_mode="r")
            y = np.load(os.path.join(self.shard_dir, f"y_{shard_id:05d}.npy"), mmap_mode="r")
            self._x_cache = {shard_id: x}  # keep one shard open to bound RAM
            self._y_cache = {shard_id: y}
        return self._x_cache[shard_id], self._y_cache[shard_id]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.df.iloc[idx]
        sid = int(r["shard_id"])
        off = int(r["offset"])
        x_sh, y_sh = self._load_shard(sid)

        x = np.asarray(x_sh[off], dtype=np.float32)
        x = np.expand_dims(x, axis=0)  # [1,H,W]
        y = int(y_sh[off])

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class TinyPhaseCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


def train_from_shards(shard_dir: str, model_dir: str, cfg: TrainConfig) -> Dict[str, float]:
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device(cfg.device)

    train_ds = PhaseShardDataset(shard_dir, split="train")
    val_ds = PhaseShardDataset(shard_dir, split="val")
    test_ds = PhaseShardDataset(shard_dir, split="test")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=pin,
        persistent_workers=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=pin,
        persistent_workers=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = TinyPhaseCNN(n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_ckpt = os.path.join(model_dir, "best_phase_cnn.pt")
    last_ckpt = os.path.join(model_dir, "last_phase_cnn.pt")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = ce(logits, y) / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * cfg.grad_accum_steps * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        # flush tail gradients if step count not divisible by grad_accum_steps
        if len(train_loader) % cfg.grad_accum_steps != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sched.step()

        tr_loss = running_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        va_loss, va_acc = evaluate(model, val_loader, device)

        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "val_acc": va_acc,
            "train_acc": tr_acc,
            "config": cfg.__dict__,
        }
        torch.save(ckpt, last_ckpt)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(ckpt, best_ckpt)

        print(
            f"[epoch {ep:03d}/{cfg.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

    # final test on best checkpoint
    best = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best["model_state"])
    te_loss, te_acc = evaluate(model, test_loader, device)

    summary = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(te_acc),
        "test_loss": float(te_loss),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
    }

    with open(os.path.join(model_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] best_val_acc={best_val_acc:.4f} | test_acc={te_acc:.4f}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colab-safe full-dataset streaming trainer")
    sub = p.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract phase maps to shards")
    p_extract.add_argument("--input-dir", required=True, help="Dataset root directory")
    p_extract.add_argument("--cache-dir", required=True, help="Directory for phase shards")
    p_extract.add_argument("--shard-size", type=int, default=2048)
    p_extract.add_argument("--image-size", type=int, default=256)
    p_extract.add_argument("--dtype", choices=["float16", "float32"], default="float16")

    p_train = sub.add_parser("train", help="Train from existing shards")
    p_train.add_argument("--shard-dir", required=True, help="Path to cache_dir/phase_shards")
    p_train.add_argument("--model-dir", required=True, help="Directory for checkpoints")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--grad-accum-steps", type=int, default=2)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p_all = sub.add_parser("all", help="Run extract + train")
    p_all.add_argument("--input-dir", required=True)
    p_all.add_argument("--cache-dir", required=True)
    p_all.add_argument("--model-dir", required=True)
    p_all.add_argument("--shard-size", type=int, default=2048)
    p_all.add_argument("--image-size", type=int, default=256)
    p_all.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p_all.add_argument("--epochs", type=int, default=100)
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--lr", type=float, default=1e-3)
    p_all.add_argument("--weight-decay", type=float, default=1e-4)
    p_all.add_argument("--num-workers", type=int, default=2)
    p_all.add_argument("--grad-accum-steps", type=int, default=2)
    p_all.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main() -> None:
    seed_everything(42)
    args = parse_args()

    if args.command == "extract":
        extract_to_shards(
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            shard_size=args.shard_size,
            image_size=args.image_size,
            out_dtype=args.dtype,
        )
        return

    if args.command == "train":
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            grad_accum_steps=args.grad_accum_steps,
            device=args.device,
        )
        train_from_shards(args.shard_dir, args.model_dir, cfg)
        return

    # all
    shard_dir = extract_to_shards(
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        shard_size=args.shard_size,
        image_size=args.image_size,
        out_dtype=args.dtype,
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device,
    )
    train_from_shards(shard_dir, args.model_dir, cfg)


if __name__ == "__main__":
    main()
