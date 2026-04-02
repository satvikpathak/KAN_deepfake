# KAN-Driven Phase-Spectrum Analysis for Deepfake Detection

A novel deepfake detection approach that leverages the **phase spectrum** of images (via 2D FFT) and **Kolmogorov-Arnold Networks (KAN)** for interpretable, parameter-efficient classification.

## Pipeline

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `stage1_phase_extraction.ipynb` | FFT phase extraction pipeline |
| 2 | `stage2_phase_analysis.ipynb` | Batch processing + KS statistical tests |
| 3 | `stage3_pca_mlp_baseline.ipynb` | PCA + MLP-Phase baseline (B3) |
| 4 | `stage4_kan_phase.ipynb` | **KAN-Phase** detector + KAN vs MLP ablation |
| 5 | `stage5_cnn_vit_baselines.ipynb` | ResNet-50 RGB/Mag (B1, B2) + ViT-B/16 (B4) |
| 6 | `stage6_ablations_robustness.ipynb` | Ablations A1-A6 + JPEG/blur robustness |
| 7 | `stage7_evaluation_interpretability.ipynb` | Final comparison + KAN interpretability |

## Running on Kaggle

1. Go to [ArtiFact Dataset](https://www.kaggle.com/datasets/awsaf49/artifact-dataset)
2. Click **New Notebook** → enable **GPU T4**
3. Import notebooks from this GitHub repo or upload manually
4. Run sequentially: Stage 1 → 2 → 3 → 4 → 5 → 6 → 7

> Stage 2 caches phase maps — all downstream stages load from cache automatically.

## Dependencies

Pre-installed on Kaggle except `pykan` (auto-installed in Stages 4/6/7):

```
numpy, pandas, opencv-python, scikit-image, scikit-learn,
matplotlib, seaborn, torch, torchvision, pykan, tqdm, scipy
```

## Dataset

**ArtiFact** (awsaf49/artifact-dataset) — multi-generator deepfake dataset with real and AI-generated images.

## Key Results

- Phase spectrum is statistically discriminative between real and fake images
- KAN achieves competitive AUC with significantly fewer parameters than CNNs
- Learned B-spline activations provide interpretable frequency-domain features
- Detection is robust to JPEG compression and generalises to unseen generators

## Colab-Safe Full-Dataset Training (32GB+) 

If the notebooks crash after Stage 2 due to RAM limits, use the streaming pipeline in
`streaming_colab_pipeline.py`. It avoids loading all phase maps into memory.

### Why this works

- **Disk-backed shards**: phase maps are written to `phase_shards/x_*.npy` + `y_*.npy`
- **Memory mapping**: training reads shards with `mmap_mode='r'`
- **Bounded DataLoader memory**: small worker count and conservative prefetching
- **AMP + grad accumulation**: stable 100-epoch training with lower VRAM usage

### Run on Colab

1. Mount Drive and point `--cache-dir` / `--model-dir` to Drive paths.
2. Install deps from `requirements.txt`.
3. Run extract once, then train (or run `all`).

```bash
python streaming_colab_pipeline.py extract \
	--input-dir /content/dataset \
	--cache-dir /content/drive/MyDrive/kan_cache \
	--shard-size 2048 \
	--image-size 256 \
	--dtype float16

python streaming_colab_pipeline.py train \
	--shard-dir /content/drive/MyDrive/kan_cache/phase_shards \
	--model-dir /content/drive/MyDrive/kan_models \
	--epochs 100 \
	--batch-size 32 \
	--grad-accum-steps 2 \
	--num-workers 2
```

### Notes

- `--epochs 100` is supported out-of-the-box.
- If Colab still runs out of memory, reduce `--batch-size` first, then `--shard-size`.
- Accuracy depends on data quality/splits/model settings; this pipeline guarantees memory safety,
	not a fixed accuracy threshold.
