# Solar Flare Forecasting

Deep learning pipeline for 24-hour M-class-or-stronger solar flare forecasting from SDO active-region image patches.

For a plain-language run guide, see [docs/experiment_guide.md](docs/experiment_guide.md).

## First Milestone

The current code supports the HMI-only binary image-classification baseline:

1. Read SDOBenchmark metadata.
2. Convert `peak_flux >= 1e-5` into label `1`, otherwise `0`.
3. Train ResNet18, EfficientNet-B0, or ConvNeXt-Tiny.
4. Evaluate accuracy, precision, recall, F1, ROC-AUC, PR-AUC, TSS, HSS, and confusion matrix.
5. Generate Grad-CAM overlays.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA GPU, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Folder Layout

```text
data/
  raw/
    sdobenchmark/
    active_region_magnetograms/
    suryabench/
  processed/
    labels/
src/
scripts/
outputs/
  checkpoints/
  logs/
  figures/
  results/
  confusion_matrices/
  gradcam_results/
```

## Usage

Put SDOBenchmark files under `data/raw/sdobenchmark/`, then run exploration:

```bash
python scripts/02_explore_dataset.py --metadata data/raw/sdobenchmark/metadata.csv
```

If you run the downloader again after the dataset is already present, it will skip re-downloading by default.

Prepare binary labels and splits:

```bash
python scripts/03_prepare_labels.py --metadata data/raw/sdobenchmark/metadata.csv --channel hmi
```

Train the first recommended model:

```bash
python scripts/05_train_efficientnet.py --epochs 30 --batch-size 16
```

Training prints detailed progress to the console, including device selection, dataset sizes, class balance, batch counts, model parameter counts, loss values, learning rate, and validation metrics for every epoch.

Evaluate the checkpoint:

```bash
python scripts/07_evaluate_model.py --checkpoint outputs/checkpoints/efficientnet_b0_best.pth
```

Evaluation prints the full metric report to the console and also saves predictions, metrics, and the confusion matrix under `outputs/`.

Generate Grad-CAM images:

```bash
python scripts/08_generate_gradcam.py --checkpoint outputs/checkpoints/efficientnet_b0_best.pth --limit 16
```

## Console Logging and Metrics

The pipeline uses console logging for the main workflow so you can see what is happening during each run without opening output files.

During training, the console shows:

- Selected device and AMP status.
- Training and validation CSV paths.
- Seed, epoch count, image size, batch size, learning rate, weight decay, threshold, and patience.
- Training and validation label distribution.
- Number of batches per epoch.
- Model name and parameter counts.
- Per-epoch `train_loss`, `val_loss`, learning rate, and validation metrics.
- Best validation metrics after training finishes.

During evaluation, the console shows:

- Checkpoint path, test CSV, model name, image size, threshold, batch size, and workers.
- Test label distribution and evaluation batch count.
- Prediction score summary.
- Full evaluation metrics.

The printed binary-classification metrics are:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- TSS
- HSS
- True negatives, false positives, false negatives, and true positives

## CPU/GPU Device Selection

By default the project uses `--device auto`. It chooses the best available PyTorch backend in this order:

```text
CUDA GPU -> Intel XPU -> Apple MPS -> CPU
```

Normal automatic run:

```bash
python scripts/05_train_efficientnet.py --device auto
```

Force CPU:

```bash
python scripts/05_train_efficientnet.py --device cpu
```

Ask for GPU/accelerator, with CPU fallback if none is available:

```bash
python scripts/05_train_efficientnet.py --device gpu
```

On Windows CPU training, use `--num-workers 0` if multiprocessing causes dataloader issues:

```bash
python scripts/05_train_efficientnet.py --device auto --batch-size 8 --num-workers 0
```

## Troubleshooting

If pandas prints a NumPy compatibility message mentioning `numexpr`, `bottleneck`, or `_ARRAY_API`, upgrade those optional pandas speedup packages:

```bash
python -m pip install --upgrade numexpr bottleneck
```

This fixes Conda environments where NumPy 2.x is installed but `numexpr` or `bottleneck` were compiled for NumPy 1.x.

If the metadata uses nonstandard column names, pass them explicitly:

```bash
python scripts/03_prepare_labels.py ^
  --metadata data/raw/sdobenchmark/metadata.csv ^
  --peak-flux-col future_peak_flux ^
  --image-col hmi_image_path
```

## Outputs

- Best checkpoints: `outputs/checkpoints/`
- Training logs: `outputs/logs/`
- Metrics and predictions: `outputs/results/`
- Confusion matrices: `outputs/confusion_matrices/`
- Grad-CAM overlays: `outputs/gradcam_results/`

Training logs are saved as CSV files, while evaluation metrics are saved as JSON files. The same key evaluation metrics are also printed directly in the console.
