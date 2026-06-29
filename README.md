# Solar Flare Forecasting

Deep learning pipeline for 24-hour M-class-or-stronger solar flare forecasting from SDO active-region image patches.

For a plain-language run guide, see [docs/experiment_guide.md](docs/experiment_guide.md).

## First Milestone

The current code supports the HMI-only binary image-classification baseline:

1. Read SDOBenchmark metadata.
2. Convert `peak_flux >= 1e-5` into label `1`, otherwise `0`.
3. Train ResNet18, EfficientNet-B0, ConvNeXt-Tiny, Swin-T, or ViT-B/16.
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
  logs/
    scripts/
  figures/
  results/
    model_comparison.csv
  pipeline_runs/
    <timestamp>_run_all/
      <timestamp>_run_all_summary.json
  experiments/
    <timestamp>_<model>_epochs<N>/
      checkpoints/
      logs/
      figures/
      results/
      data/
  evaluations/
    <timestamp>_eval_<model>_epochs<N>_<checkpoint>/
      logs/
      results/
      confusion_matrices/
  gradcam_results/
    <timestamp>_gradcam_<model>_epochs<N>_<checkpoint>/
      logs/
      images/
```

## Usage

To run the full experiment pipeline with one command:

```bash
python scripts/00_run_all.py --epochs 30 --batch-size 16 --device auto
```

By default this runs dataset checks, label preparation, training for ResNet18, EfficientNet-B0, ConvNeXt-Tiny, Swin-T, and ViT-B/16, threshold optimization, evaluation, and Grad-CAM. It skips dataset download and preprocessing outputs when they already exist.

Useful full-run options:

```bash
python scripts/00_run_all.py --models efficientnet_b0 convnext_tiny --epochs 30
python scripts/00_run_all.py --models all --skip-gradcam
python scripts/00_run_all.py --skip-threshold --skip-gradcam --epochs 5
```

Manual run order is below if you want to run each step separately.

0. Run everything from one file.

```bash
python scripts/00_run_all.py --epochs 30 --batch-size 16
```

1. Download or prepare the raw dataset.

If SDOBenchmark is already present under `data/raw/sdobenchmark/`, this script skips the download. Use `--force` only when you intentionally want to download again.

```bash
python scripts/01_download_data.py --url <DATASET_URL> --output data/raw/sdobenchmark/sdobenchmark.zip --extract
```

2. Explore the dataset.

Exploration outputs are reused if they already exist. Add `--force` to recreate them.

```bash
python scripts/02_explore_dataset.py --metadata data/raw/sdobenchmark/metadata.csv
```

3. Prepare binary labels and splits.

If `train_labels.csv`, `val_labels.csv`, and `test_labels.csv` already exist, preprocessing is skipped. Add `--force` to recreate them.

```bash
python scripts/03_prepare_labels.py --metadata data/raw/sdobenchmark/metadata.csv --channel hmi
```

4. Train one or more models.

Each training run creates a new timestamped folder under `outputs/experiments/`.

```bash
python scripts/04_train_resnet18.py --epochs 30 --batch-size 16
python scripts/05_train_efficientnet.py --epochs 30 --batch-size 16
python scripts/06_train_convnext.py --epochs 30 --batch-size 8
python scripts/07_train_swin_transformer.py --epochs 30 --batch-size 8
python scripts/08_train_vit_transformer.py --epochs 30 --batch-size 8
```

Training prints detailed progress to the console, including device selection, dataset sizes, class balance, batch counts, model parameter counts, loss values, learning rate, and validation metrics for every epoch.

5. Optionally tune the decision threshold on the validation set.

```bash
python scripts/09_optimize_threshold.py --checkpoint outputs/experiments/<run_name>/checkpoints/<run_name>_best.pth
```

6. Evaluate the checkpoint.

Evaluation creates a timestamped folder under `outputs/evaluations/`. The saved metrics JSON records the exact checkpoint path, checkpoint file, checkpoint epoch, and training run folder.

```bash
python scripts/10_evaluate_model.py --checkpoint outputs/experiments/<run_name>/checkpoints/<run_name>_best.pth
```

7. Generate Grad-CAM images.

```bash
python scripts/11_generate_gradcam.py --checkpoint outputs/experiments/<run_name>/checkpoints/<run_name>_best.pth --limit 16
```

## Console Logging and Metrics

The pipeline uses console logging for the main workflow so you can see what is happening during each run without opening output files.

Every standalone script saves a timestamped log file in `outputs/logs/scripts/`. Training, evaluation, threshold tuning, and Grad-CAM also save run-specific logs inside their timestamped run folders.

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

Evaluation metrics files also include the checkpoint path and training run metadata, so you can always tell which weights produced each result.

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

Training run folders contain:

- `checkpoints/<run_name>_best.pth`
- `checkpoints/<run_name>_last.pth`
- `logs/<run_name>_train.log`
- `logs/<run_name>_training_log.csv`
- `figures/<run_name>_training_curves.png`
- `results/<run_name>_training_summary.json`
- `data/<run_name>_train_labels.csv` and `data/<run_name>_val_labels.csv`

Evaluation run folders contain:

- `logs/<eval_run_name>_evaluation.log`
- `results/<eval_run_name>_metrics.json`
- `results/<eval_run_name>_predictions.csv`
- `results/model_comparison.csv`
- `confusion_matrices/<eval_run_name>_confusion_matrix.png`

The global comparison table is also updated at `outputs/results/model_comparison.csv`.
