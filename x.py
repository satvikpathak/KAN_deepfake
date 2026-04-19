#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
 KAN-Based Deepfake Detection — Production v2 (x.ipynb)
 ─────────────────────────────────────────────────────────
 Dataset  : DeepDetect-2025 (full ~112 k images)
 Model    : Dual-Channel PhaseKAN with Residual Conv Stem
 Target   : Binary classification — Real (0) vs Fake (1)
 Hardware : Google Colab T4 / L4 GPU — AMP enabled
==============================================================================
"""

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Environment Setup
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess, sys

def _pip(pkg: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

try:
    import kagglehub
except ImportError:
    _pip("kagglehub")
    import kagglehub

try:
    import sklearn
except ImportError:
    _pip("scikit-learn")

import os, random, time, json, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler          # mixed precision
import torchvision.transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── deterministic seeds ──────────────────────────────────────────────────────
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"AMP     : enabled (float16)")
print(f"Seed    : {SEED}")
print("Environment ready ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Download Full Dataset
# ═══════════════════════════════════════════════════════════════════════════════

dataset_path = kagglehub.dataset_download("ayushmandatta1/deepdetect-2025")
DATASET_ROOT = Path(dataset_path)
print(f"Dataset root: {DATASET_ROOT}")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def discover_images(root: Path) -> List[Tuple[str, int]]:
    """Walk dataset tree and infer labels.  0 = Real, 1 = Fake."""
    samples: List[Tuple[str, int]] = []
    for dirpath, _, filenames in os.walk(root):
        dp_lower = dirpath.lower()
        if "real" in dp_lower:
            label = 0
        elif "fake" in dp_lower or "ai" in dp_lower or "synthetic" in dp_lower:
            label = 1
        else:
            generators = ["midjourney", "sd3", "stable", "dall", "dalle"]
            if any(g in dp_lower for g in generators):
                label = 1
            else:
                continue
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXTS:
                samples.append((os.path.join(dirpath, fn), label))
    return samples


all_samples = discover_images(DATASET_ROOT)
random.shuffle(all_samples)

# ── balance classes (use ALL images — no cap) ────────────────────────────────
reals = [(p, l) for p, l in all_samples if l == 0]
fakes = [(p, l) for p, l in all_samples if l == 1]
n_per_class = min(len(reals), len(fakes))
reals = reals[:n_per_class]
fakes = fakes[:n_per_class]
all_samples = reals + fakes
random.shuffle(all_samples)

print(f"Total images : {len(all_samples)}")
print(f"  Real : {len(reals)}")
print(f"  Fake : {len(fakes)}")
print("Full dataset loaded ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Dataset + Dual-Channel GPU Extractor
# ═══════════════════════════════════════════════════════════════════════════════

CROP_SIZE = 224


class SpatialDataset(Dataset):
    """Returns raw grayscale spatial tensors.  FFT is done on GPU."""

    def __init__(self, samples, crop_size=CROP_SIZE, is_train=False):
        self.samples   = samples
        self.crop_size = crop_size
        if is_train:
            self.tf = T.Compose([
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.tf = T.CenterCrop(crop_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (self.crop_size, self.crop_size), 0)

        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            nw, nh = max(w, self.crop_size), max(h, self.crop_size)
            padded = Image.new("L", (nw, nh), 0)
            padded.paste(img, ((nw - w) // 2, (nh - h) // 2))
            img = padded

        img = self.tf(img)
        return T.ToTensor()(img), torch.tensor(label, dtype=torch.float32)


def extract_dual_fft_gpu(batch: torch.Tensor) -> torch.Tensor:
    """
    (B,1,H,W) spatial → (B,2,H,W) [phase, log-magnitude] on GPU.
    """
    x = batch.squeeze(1)
    fft  = torch.fft.fft2(x)
    fftc = torch.fft.fftshift(fft, dim=(-2, -1))

    # Phase: [-π, π] → [0, 1]
    phase = (torch.angle(fftc) + torch.pi) / (2 * torch.pi)

    # Magnitude: log-scaled, per-sample min-max → [0, 1]
    mag = torch.log(torch.abs(fftc) + 1e-8)
    lo  = mag.amin(dim=(-2, -1), keepdim=True)
    hi  = mag.amax(dim=(-2, -1), keepdim=True)
    mag = (mag - lo) / (hi - lo + 1e-8)

    return torch.stack([phase, mag], dim=1)


# Quick sanity check
print("Sanity check ...")
_ds = SpatialDataset(all_samples[:1], is_train=False)
_s, _l = _ds[0]
print(f"  Spatial : {_s.shape}, [{_s.min():.3f}, {_s.max():.3f}]")
if torch.cuda.is_available():
    _d = extract_dual_fft_gpu(_s.unsqueeze(0).to(DEVICE))
    print(f"  Dual    : {_d.shape}, phase [{_d[:,0].min():.3f},{_d[:,0].max():.3f}], mag [{_d[:,1].min():.3f},{_d[:,1].max():.3f}]")
    del _d; torch.cuda.empty_cache()
print("Dataset + GPU extractor OK ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — DataLoaders
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
BATCH_SIZE  = 64
NUM_WORKERS = 0

n_total = len(all_samples)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

indices = list(range(n_total))
random.shuffle(indices)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

train_full = SpatialDataset(all_samples, is_train=True)
eval_full  = SpatialDataset(all_samples, is_train=False)

train_ds = Subset(train_full, train_idx)
val_ds   = Subset(eval_full,  val_idx)
test_ds  = Subset(eval_full,  test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train : {len(train_ds)}")
print(f"Val   : {len(val_ds)}")
print(f"Test  : {len(test_ds)}")
print("DataLoaders ready ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 4.5 — IEEE Methodology Visualiser
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📐 Generating IEEE methodology figure ...")


def plot_fft_transformations(dataset):
    real_idx = fake_idx = -1
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label.item() == 0 and real_idx == -1: real_idx = i
        if label.item() == 1 and fake_idx == -1: fake_idx = i
        if real_idx != -1 and fake_idx != -1: break

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Dual-Channel FFT Extraction for KAN Input",
                 fontsize=16, fontweight="bold", y=0.95)

    for row, (idx, name) in enumerate(
        [(real_idx, "Real Image"), (fake_idx, "AI-Generated (Fake)")]
    ):
        sp, _ = dataset[idx]
        img = sp.squeeze(0).numpy()
        fft_s = np.fft.fftshift(np.fft.fft2(img))

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"{name}\nSpatial Domain", fontsize=12)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.log(np.abs(fft_s) + 1e-8), cmap="viridis")
        axes[row, 1].set_title("Ch 2: Magnitude\n[Energy]", fontsize=12)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(np.angle(fft_s), cmap="twilight")
        axes[row, 2].set_title("Ch 1: Phase\n[Structure]", fontsize=12)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("dual_channel_fft_ieee.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved → dual_channel_fft_ieee.png ✅")


plot_fft_transformations(val_ds)


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — KAN Architecture (Residual Conv Stem + Wide KAN Head)
# ═══════════════════════════════════════════════════════════════════════════════

class BSplineActivation(nn.Module):
    """Learnable univariate activation via B-spline basis."""

    def __init__(self, in_features: int, num_knots: int = 16, spline_order: int = 3):
        super().__init__()
        self.in_features  = in_features
        self.num_knots    = num_knots
        self.spline_order = spline_order
        n_bases = num_knots + spline_order - 1
        self.coeff = nn.Parameter(torch.randn(in_features, n_bases) * 0.1)
        grid = torch.linspace(-1.0, 1.0, num_knots + 2 * spline_order)
        self.register_buffer("grid", grid)

    def _b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        n_bases = self.num_knots + self.spline_order - 1
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        bases = bases[..., :n_bases + self.spline_order]

        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1] - 1
            left_num  = x - grid[:n].unsqueeze(0).unsqueeze(0)
            left_den  = (grid[k:k+n] - grid[:n]).unsqueeze(0).unsqueeze(0)
            right_num = grid[k+1:k+1+n].unsqueeze(0).unsqueeze(0) - x
            right_den = (grid[k+1:k+1+n] - grid[1:1+n]).unsqueeze(0).unsqueeze(0)
            left  = left_num  / (left_den  + 1e-8) * bases[..., :n]
            right = right_num / (right_den + 1e-8) * bases[..., 1:n+1]
            bases = left + right
        return bases[..., :self.coeff.shape[1]]

    def forward(self, x):
        x_c = torch.clamp(x, -1.0, 1.0)
        basis = self._b_spline_basis(x_c)
        return (basis * self.coeff.unsqueeze(0)).sum(dim=-1)


class KANLinear(nn.Module):
    """KAN layer: learnable spline activation → linear + residual path."""

    def __init__(self, in_f: int, out_f: int, num_knots: int = 16):
        super().__init__()
        self.spline      = BSplineActivation(in_f, num_knots=num_knots)
        self.linear      = nn.Linear(in_f, out_f)
        self.base_linear = nn.Linear(in_f, out_f)
        self.base_act    = nn.SiLU()

    def forward(self, x):
        return self.linear(self.spline(x)) + self.base_linear(self.base_act(x))


class ResBlock(nn.Module):
    """Conv residual block with skip connection."""

    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch_out)
        self.act   = nn.GELU()

        # Skip connection: match dimensions if needed
        if stride != 1 or ch_in != ch_out:
            self.skip = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class PhaseKAN(nn.Module):
    """
    Dual-Channel PhaseKAN v2 — Residual Conv Stem + Wide KAN Head.

    Input:  (B, 2, 224, 224) — [phase, magnitude]
    Output: (B, 1) — logit

    Conv Stem (Residual):
      2→64 (s2) → 64→128 (s1) → 128→128 (s2) → 128→256 (s1) → 256→256 (s2)
      = 224→112→112→56→56→28  →  GAP → 256-dim

    KAN Head: 256→256→128→64→1  (16-knot B-splines)
    """

    def __init__(self, in_ch=2, kan_hidden=None, num_knots=16, dropout=0.3):
        super().__init__()
        if kan_hidden is None:
            kan_hidden = [256, 128, 64]

        self.stem = nn.Sequential(
            ResBlock(in_ch, 64,  stride=2),    # 224→112
            ResBlock(64,    128, stride=1),     # 112→112  (preserve detail)
            ResBlock(128,   128, stride=2),     # 112→56
            ResBlock(128,   256, stride=1),     # 56→56    (preserve detail)
            ResBlock(256,   256, stride=2),     # 56→28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        conv_dim = 256

        dims = [conv_dim] + kan_hidden + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i+1], num_knots=num_knots))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        h = self.stem(x)
        h = self.pool(h).flatten(1)
        return self.head(h)


model = PhaseKAN().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print(model)
print("PhaseKAN v2 ready ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — Training Loop (AMP + OneCycleLR + Label Smoothing)
# ═══════════════════════════════════════════════════════════════════════════════

NUM_EPOCHS    = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-3
GRAD_CLIP     = 1.0
PATIENCE      = 12
LABEL_SMOOTH  = 0.05       # reduce overconfidence

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([1.0], device=DEVICE),
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
    pct_start=0.1, anneal_strategy="cos",
)
scaler = GradScaler("cuda")


def smooth_labels(labels, smoothing=LABEL_SMOOTH):
    """Apply label smoothing: 0 → ε, 1 → 1-ε."""
    return labels * (1 - smoothing) + 0.5 * smoothing


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    n_correct = n_total = 0

    for spatial, labels in tqdm(loader, desc="  train", leave=False):
        spatial = spatial.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True).unsqueeze(1)

        # GPU FFT
        dual = extract_dual_fft_gpu(spatial)

        # Label smoothing
        smooth = smooth_labels(labels)

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(dual)
            loss   = criterion(logits, smooth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * spatial.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        n_correct += (preds == labels).sum().item()
        n_total   += labels.size(0)

    return running_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for spatial, labels in tqdm(loader, desc="  eval ", leave=False):
        spatial = spatial.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True).unsqueeze(1)

        dual = extract_dual_fft_gpu(spatial)

        with autocast("cuda"):
            logits = model(dual)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * spatial.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    n_total    = len(all_labels)
    preds      = (all_probs >= 0.5).astype(float)
    return (running_loss / n_total,
            accuracy_score(all_labels, preds),
            all_probs, all_labels)


# ── Training ─────────────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")
patience_ctr  = 0
best_state    = None

print(f"\n{'='*60}")
print(f" Training — {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
print(f" AMP=float16 | LabelSmooth={LABEL_SMOOTH} | Patience={PATIENCE}")
print(f"{'='*60}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    tr_loss, tr_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE
    )
    vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

    dt = time.time() - t0
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    star = ""
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_ctr  = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        star = " ★"
    else:
        patience_ctr += 1

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} │ "
        f"train {tr_loss:.4f} / {tr_acc:.4f} │ "
        f"val {vl_loss:.4f} / {vl_acc:.4f} │ "
        f"{dt:.1f}s{star}"
    )

    if patience_ctr >= PATIENCE:
        print(f"\n⏹  Early stopping at epoch {epoch} (patience={PATIENCE})")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(DEVICE)
print("\nTraining complete ✅  (best model restored)")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — Test Metrics
# ═══════════════════════════════════════════════════════════════════════════════

test_loss, test_acc, test_probs, test_labels = evaluate(
    model, test_loader, criterion, DEVICE
)
test_preds = (test_probs >= 0.5).astype(float)

precision = precision_score(test_labels, test_preds, zero_division=0)
recall    = recall_score(test_labels, test_preds, zero_division=0)
f1        = f1_score(test_labels, test_preds, zero_division=0)
roc_auc   = roc_auc_score(test_labels, test_probs)

print(f"\n{'='*60}")
print(f" Test-Set Results")
print(f"{'='*60}")
print(f"  Loss      : {test_loss:.4f}")
print(f"  Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"{'='*60}\n")
print(classification_report(
    test_labels, test_preds,
    target_names=["Real", "Fake"], digits=4
))


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 Generating visualisations ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC
fpr, tpr, _ = roc_curve(test_labels, test_probs)
axes[0].plot(fpr, tpr, color="royalblue", lw=2,
             label=f"KAN (AUC = {roc_auc:.4f})")
axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"]).plot(
    ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Confusion Matrix", fontweight="bold")

# Learning Curves
ep = list(range(1, len(history["train_loss"]) + 1))
axes[2].plot(ep, history["train_loss"], label="Train Loss", lw=2)
axes[2].plot(ep, history["val_loss"],   label="Val Loss",   lw=2)
ax2 = axes[2].twinx()
ax2.plot(ep, history["train_acc"], "--", label="Train Acc", lw=1.5, color="green")
ax2.plot(ep, history["val_acc"],   "--", label="Val Acc",   lw=1.5, color="red")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss"); ax2.set_ylabel("Accuracy")
axes[2].set_title("Learning Curves", fontweight="bold")
lines1, lab1 = axes[2].get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
axes[2].legend(lines1+lines2, lab1+lab2, fontsize=9, loc="center right")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("kan_deepfake_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Results saved → kan_deepfake_results.png ✅")


# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 9 — Save Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = Path("runs/kan_deepfake_v2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": len(history["train_loss"]),
    "best_val_loss": best_val_loss,
    "config": {
        "crop_size": CROP_SIZE, "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE, "wd": WEIGHT_DECAY,
        "label_smooth": LABEL_SMOOTH, "seed": SEED,
    },
}, SAVE_DIR / "best_phase_kan_v2.pt")
print(f"Checkpoint → {SAVE_DIR / 'best_phase_kan_v2.pt'}")

summary = {
    "dataset": "DeepDetect-2025 (full)",
    "model": "PhaseKAN v2 (Residual stem + Dual-channel + Wide KAN)",
    "parameters": n_params,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "test_samples": len(test_ds),
    "epochs_trained": len(history["train_loss"]),
    "test_metrics": {
        "accuracy": round(test_acc, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    },
    "history": history,
}
with open(SAVE_DIR / "summary_v2.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary  → {SAVE_DIR / 'summary_v2.json'}")
print("\n🏁 Pipeline complete.")
