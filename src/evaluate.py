"""Evaluate trained flare classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from . import config
from .dataset import SolarFlareImageDataset, build_transforms, make_dataloader
from .metrics import binary_classification_metrics, log_binary_metrics, save_confusion_matrix
from .models import get_model
from .utils import build_run_name, ensure_dir, get_device, get_logger, save_json, setup_logging, timestamp

logger = get_logger(__name__)


def _label_distribution(dataset: SolarFlareImageDataset, label_col: str) -> str:
    labels = dataset.frame[label_col].astype(int)
    counts = labels.value_counts().sort_index()
    total = max(1, len(labels))
    no_flare = int(counts.get(0, 0))
    flare = int(counts.get(1, 0))
    return (
        f"no_flare={no_flare} ({no_flare / total:.2%}), "
        f"flare={flare} ({flare / total:.2%})"
    )


def _score_summary(scores: list[float]) -> str:
    if not scores:
        return "no scores"
    return f"min={min(scores):.4f} mean={sum(scores) / len(scores):.4f} max={max(scores):.4f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _checkpoint_metadata(checkpoint: object, checkpoint_path: str | Path) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_file": Path(checkpoint_path).name,
        }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_file": Path(checkpoint_path).name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_score": checkpoint.get("best_score"),
        "training_run_name": checkpoint.get("run_name"),
        "training_run_dir": checkpoint.get("run_dir"),
        "training_created_at": checkpoint.get("created_at"),
        "training_epochs_requested": checkpoint.get("epochs_requested"),
        "training_log": checkpoint.get("training_log"),
        "training_curves": checkpoint.get("training_curves"),
    }


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    config.ensure_project_dirs()
    created_at = args.run_timestamp or timestamp()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    model_name = args.model or checkpoint_dict.get("model_name", "efficientnet_b0")
    image_size = args.image_size or checkpoint_dict.get("image_size", config.IMAGE_SIZE)
    checkpoint_epoch = checkpoint_dict.get("epoch", "unknown")
    checkpoint_stem = Path(args.checkpoint).stem
    run_name = args.run_name or build_run_name(
        model_name,
        epochs=checkpoint_epoch,
        run_type="eval",
        created_at=created_at,
        suffix=checkpoint_stem,
    )
    run_dir = ensure_dir(args.run_dir or (Path(args.output_root) / run_name))
    log_dir = ensure_dir(run_dir / "logs")
    result_dir = ensure_dir(args.result_dir or (run_dir / "results"))
    confusion_dir = ensure_dir(args.confusion_dir or (run_dir / "confusion_matrices"))
    log_path = log_dir / f"{run_name}_evaluation.log"

    setup_logging(log_file=log_path)
    logger.info("Starting evaluation")
    logger.info("Evaluation run name: %s", run_name)
    logger.info("Evaluation run directory: %s", run_dir)
    logger.info("Using device: %s", device)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Test CSV: %s", args.csv)
    logger.info("Model: %s", model_name)
    logger.info("Image size: %s", image_size)
    logger.info("Evaluation threshold: %.4f", args.threshold)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Number of workers: %d", args.num_workers)
    checkpoint_info = _checkpoint_metadata(checkpoint, args.checkpoint)
    save_json(
        {
            "run_name": run_name,
            "run_type": "evaluation",
            "created_at": created_at,
            "run_dir": str(run_dir),
            "log_file": str(log_path),
            "result_dir": str(result_dir),
            "confusion_dir": str(confusion_dir),
            "checkpoint": checkpoint_info,
            "args": _json_ready(vars(args)),
        },
        run_dir / f"{run_name}_evaluation_config.json",
    )

    dataset = SolarFlareImageDataset(
        args.csv,
        image_root=args.image_root,
        transform=build_transforms(image_size, train=False),
        image_col=args.image_col,
        path_col=args.path_col,
        label_col=args.label_col,
        channel=args.channel,
        return_path=True,
    )
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("Loaded test dataset with %d samples", len(dataset))
    logger.info("Test label distribution: %s", _label_distribution(dataset, args.label_col))
    logger.info("Evaluation batches: %d", len(loader))

    model = get_model(model_name, pretrained=False).to(device)
    state_dict = checkpoint_dict.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded checkpoint weights and set model to evaluation mode")

    y_true: list[int] = []
    y_score: list[float] = []
    paths: list[str] = []

    with torch.no_grad():
        for images, labels, batch_paths in tqdm(loader, desc="evaluate"):
            images = images.to(device, non_blocking=True)
            logits = model(images).view(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            y_score.extend(scores)
            y_true.extend(labels.numpy().astype(int).tolist())
            paths.extend(batch_paths)

    metrics = binary_classification_metrics(y_true, y_score, threshold=args.threshold)
    logger.info("Collected predictions for %d samples", len(y_true))
    logger.info("Prediction score summary: %s", _score_summary(y_score))
    log_binary_metrics(logger, metrics, title="Evaluation metrics")

    predictions = pd.DataFrame(
        {
            "image_path": paths,
            "label": y_true,
            "flare_probability": y_score,
            "prediction": [int(score >= args.threshold) for score in y_score],
        }
    )
    predictions_path = result_dir / f"{run_name}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    metrics_path = result_dir / f"{run_name}_metrics.json"
    metrics_payload = {
        "run_name": run_name,
        "run_type": "evaluation",
        "created_at": created_at,
        "model_name": model_name,
        "threshold": args.threshold,
        "csv": str(args.csv),
        "predictions": str(predictions_path),
        "confusion_matrix": str(confusion_dir / f"{run_name}_confusion_matrix.png"),
        "run_dir": str(run_dir),
        **checkpoint_info,
        **{key: float(value) if isinstance(value, float) else value for key, value in metrics.items()},
    }
    save_json(metrics_payload, metrics_path)
    logger.info("Saved metrics to %s", metrics_path)

    matrix_path = confusion_dir / f"{run_name}_confusion_matrix.png"
    save_confusion_matrix(y_true, y_score, matrix_path, threshold=args.threshold, title=f"{model_name} confusion matrix")
    logger.info("Saved confusion matrix to %s", matrix_path)

    comparison_path = result_dir / "model_comparison.csv"
    row = {
        "model": model_name,
        "evaluation_run_name": run_name,
        "evaluation_run_dir": str(run_dir),
        **checkpoint_info,
        **metrics,
    }
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    else:
        comparison = pd.DataFrame([row])
    comparison.to_csv(comparison_path, index=False)
    logger.info("Updated model comparison table at %s", comparison_path)

    global_comparison_path = config.RESULT_DIR / "model_comparison.csv"
    ensure_dir(global_comparison_path.parent)
    if global_comparison_path.exists():
        global_comparison = pd.read_csv(global_comparison_path)
        global_comparison = pd.concat([global_comparison, pd.DataFrame([row])], ignore_index=True)
    else:
        global_comparison = pd.DataFrame([row])
    dedupe_columns = ["model", "checkpoint", "evaluation_run_name"]
    global_comparison = global_comparison.drop_duplicates(subset=dedupe_columns, keep="last")
    global_comparison.to_csv(global_comparison_path, index=False)
    logger.info("Updated global model comparison table at %s", global_comparison_path)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained solar flare classifier.")
    parser.add_argument("--csv", default=str(config.LABELS_DIR / "test_labels.csv"))
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, choices=(*config.MODEL_NAMES, None))
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--path-col", default=config.IMAGE_PATH_COLUMN)
    parser.add_argument("--label-col", default=config.LABEL_COLUMN)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    parser.add_argument("--output-root", default=str(config.EVALUATION_DIR))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-timestamp", default=None)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--confusion-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
