# Solar Flare Experiment Guide

This project is for predicting strong solar flares from SDO active-region images.
The main idea is simple:

1. Get the image data and metadata.
2. Turn the flare strength into a binary label.
3. Train image classification models.
4. Test the models.
5. Save graphs, confusion matrices, and explanation images for the paper.

This guide explains what each step does, which files are required first, and which scripts are only for later experiments.

## Why We Do This

Solar flares can affect satellites, GPS, communication systems, and power grids.
In this project, we use machine learning to learn patterns in solar active-region images and predict whether a strong flare will happen in the next 24 hours.

The first version of the project is the simplest and most important one:

`HMI magnetogram images -> binary flare label -> deep learning classifier`

If this baseline works well, we can later add more channels and more advanced models.

## What Is Compulsory Before Training

These steps are required before any model training:

1. Download the SDOBenchmark HMI data.
2. Make sure the image folders are in `data/raw/sdobenchmark/`.
3. Make sure `training/meta_data.csv` and `test/meta_data.csv` exist.
4. Create the processed label CSV files.
5. Check that the image paths are valid.

If any of these are missing, training will fail or give wrong results.

## Required Files Now

These are the only files we need for the first milestone:

| File or Data | Why it is needed |
| --- | --- |
| `data/raw/sdobenchmark/training/meta_data.csv` | Training metadata with flare values |
| `data/raw/sdobenchmark/test/meta_data.csv` | Test metadata with flare values |
| HMI magnetogram images | Main input for the classifier |
| `data/processed/labels/train_labels.csv` | Training labels and image paths |
| `data/processed/labels/val_labels.csv` | Validation labels and image paths |
| `data/processed/labels/test_labels.csv` | Final test labels and image paths |

We do not need AIA channels, SuryaBench CSV, or external validation data yet.

## What Each Script Does

### `scripts/00_run_all.py`

This is the all-in-one runner.
It runs dataset checks, exploration, label preparation, model training, threshold optimization, evaluation, and Grad-CAM from one file.

Default models:

- ResNet18
- EfficientNet-B0
- ConvNeXt-Tiny
- Swin-T
- ViT-B/16

What it generates:

- `outputs/pipeline_runs/<pipeline_run_name>/<pipeline_run_name>_summary.json`
- timestamped experiment folders under `outputs/experiments/`
- timestamped evaluation folders under `outputs/evaluations/`
- timestamped Grad-CAM folders under `outputs/gradcam_results/`
- a pipeline log under `outputs/logs/scripts/`

Example:

```bash
python scripts/00_run_all.py --epochs 30 --batch-size 16 --device auto
```

### `scripts/01_download_data.py`

This script downloads the dataset archive if you still need it.
It is only for getting the raw data onto your machine.

What it generates:

- raw dataset files in `data/raw/sdobenchmark/`
- a timestamped script log in `outputs/logs/scripts/`

If the dataset or archive is already present, the script skips downloading unless `--force` is used.

### `scripts/02_explore_dataset.py`

This script checks the dataset before training.
It helps us see how many samples we have, how the labels are distributed, and whether any image paths are missing or broken.

What it generates:

- dataset summary JSON
- class distribution graph
- a timestamped script log in `outputs/logs/scripts/`

If the exploration outputs already exist, the script reuses them unless `--force` is used.

Why it matters:

- it tells us whether the dataset is imbalanced
- it helps us catch missing files early

### `scripts/03_prepare_labels.py`

This is a compulsory step before training.
It converts the flare strength value `peak_flux` into a binary label:

```python
if peak_flux >= 1e-5:
    label = 1
else:
    label = 0
```

For this project:

- `1` means strong flare
- `0` means no strong flare

What it generates:

- `data/processed/labels/train_labels.csv`
- `data/processed/labels/val_labels.csv`
- `data/processed/labels/test_labels.csv`
- a timestamped script log in `outputs/logs/scripts/`

If all three label CSV files already exist, preprocessing is skipped unless `--force` is used.

Why it matters:

- training needs binary labels
- the dataset metadata is not directly ready for classification

### `scripts/04_train_resnet18.py`

This trains the ResNet18 baseline.
It is useful as a simple comparison model.

What it generates:

- `outputs/experiments/<run_name>/checkpoints/<run_name>_best.pth`
- `outputs/experiments/<run_name>/checkpoints/<run_name>_last.pth`
- `outputs/experiments/<run_name>/logs/<run_name>_train.log`
- `outputs/experiments/<run_name>/logs/<run_name>_training_log.csv`
- `outputs/experiments/<run_name>/figures/<run_name>_training_curves.png`
- `outputs/experiments/<run_name>/results/<run_name>_training_summary.json`

### `scripts/05_train_efficientnet.py`

This trains the EfficientNet-B0 baseline.
This is the main baseline we recommend first because it is light and strong.

What it generates:

- the same timestamped experiment folder structure as ResNet18

### `scripts/06_train_convnext.py`

This trains the ConvNeXt-Tiny baseline.
It is a stronger modern CNN comparison model.

What it generates:

- the same timestamped experiment folder structure as ResNet18

### `scripts/07_train_swin_transformer.py`

This trains the Swin-T transformer baseline.

What it generates:

- the same timestamped experiment folder structure as ResNet18

### `scripts/08_train_vit_transformer.py`

This trains the ViT-B/16 transformer baseline.

What it generates:

- the same timestamped experiment folder structure as ResNet18

### `scripts/09_optimize_threshold.py`

This searches validation thresholds for metrics such as TSS, HSS, F1, precision, or recall.

What it generates:

- a timestamped threshold run under `outputs/evaluations/`
- threshold sweep CSV
- best-threshold JSON with the source checkpoint path

### `scripts/10_evaluate_model.py`

This script loads a trained checkpoint and evaluates it on the test set.
It calculates the paper metrics and saves the confusion matrix.

What it generates:

- `outputs/evaluations/<eval_run_name>/logs/<eval_run_name>_evaluation.log`
- `outputs/evaluations/<eval_run_name>/results/<eval_run_name>_metrics.json`
- `outputs/evaluations/<eval_run_name>/results/<eval_run_name>_predictions.csv`
- `outputs/evaluations/<eval_run_name>/confusion_matrices/<eval_run_name>_confusion_matrix.png`
- `outputs/results/model_comparison.csv`

Why it matters:

- this gives the final numbers for the paper
- this is where we compare all models fairly
- each metrics JSON records the exact checkpoint used for evaluation

### `scripts/11_generate_gradcam.py`

This creates Grad-CAM images.
Grad-CAM shows which part of the solar image influenced the prediction.

What it generates:

- `outputs/gradcam_results/<gradcam_run_name>/logs/<gradcam_run_name>_gradcam.log`
- `outputs/gradcam_results/<gradcam_run_name>/images/*.png`

Why it matters:

- these images are useful in a journal paper
- they help explain the model instead of only reporting numbers

### `scripts/12_train_multichannel_model.py`

This is a placeholder for the multi-channel experiment.
It is not a real experiment yet.

### `scripts/13_train_temporal_model.py`

This is a placeholder for the temporal experiment.
It is not a real experiment yet.

### `scripts/14_external_validation.py`

This is a placeholder for external validation.
It is not a real experiment yet.

## What Must Happen First

Before running any training script, do this in order:

1. Make sure the raw SDOBenchmark HMI files are in place.
2. Run `scripts/03_prepare_labels.py`.
3. Confirm the `train_labels.csv`, `val_labels.csv`, and `test_labels.csv` files were created.
4. Confirm the image paths are valid.
5. Train one model, then move to the next.

If you skip label preparation, the training scripts will not have the correct targets.

## Recommended Run Order For The Paper

For the journal paper, use this order:

The simplest option is:

1. `scripts/00_run_all.py`

If you want to run each step manually, use:

1. `scripts/02_explore_dataset.py`
2. `scripts/03_prepare_labels.py`
3. `scripts/04_train_resnet18.py`
4. `scripts/05_train_efficientnet.py`
5. `scripts/06_train_convnext.py`
6. `scripts/07_train_swin_transformer.py`
7. `scripts/08_train_vit_transformer.py`
8. `scripts/09_optimize_threshold.py` if you want a tuned threshold
9. `scripts/10_evaluate_model.py`
10. `scripts/11_generate_gradcam.py`

This gives you:

- baseline model comparison
- test-set performance
- confusion matrices
- Grad-CAM figures
- paper-ready result tables
- timestamped folders that keep each experiment separate

## Main Outputs For The Paper

The main files you will use in the paper are:

- `outputs/results/model_comparison.csv`
- `outputs/evaluations/<eval_run_name>/confusion_matrices/*.png`
- `outputs/gradcam_results/<gradcam_run_name>/images/*.png`
- `outputs/experiments/<run_name>/logs/*.csv`
- `outputs/experiments/<run_name>/checkpoints/*.pth`
- `outputs/experiments/<run_name>/figures/*_training_curves.png`

## Easy Summary

If you want the shortest possible answer, the compulsory path is:

1. Get SDOBenchmark HMI data
2. Run label preparation
3. Train ResNet18, EfficientNet-B0, ConvNeXt-Tiny, Swin-T, and ViT-B/16 as needed
4. Optionally optimize the threshold
5. Evaluate all models
6. Save confusion matrices and Grad-CAM images

That is the baseline experiment set for the first journal results.
