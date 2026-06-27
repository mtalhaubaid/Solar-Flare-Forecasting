"""Evaluate trained flare classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from . import config
from .dataset import SolarFlareImageDataset, build_transforms, make_dataloader
from .metrics import binary_classification_metrics, save_confusion_matrix
from .models import get_model
from .utils import ensure_dir, get_device, get_logger, save_json, setup_logging

logger = get_logger(__name__)


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    setup_logging()
    logger.info("Starting evaluation")
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model_name", "efficientnet_b0")
    image_size = args.image_size or checkpoint.get("image_size", config.IMAGE_SIZE)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Test CSV: %s", args.csv)

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

    model = get_model(model_name, pretrained=False).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

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
    result_dir = ensure_dir(args.result_dir)
    confusion_dir = ensure_dir(args.confusion_dir)

    predictions = pd.DataFrame(
        {
            "image_path": paths,
            "label": y_true,
            "flare_probability": y_score,
            "prediction": [int(score >= args.threshold) for score in y_score],
        }
    )
    predictions_path = result_dir / f"{model_name}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    metrics_path = result_dir / f"{model_name}_metrics.json"
    save_json({key: float(value) if isinstance(value, float) else value for key, value in metrics.items()}, metrics_path)
    logger.info("Saved metrics to %s", metrics_path)

    matrix_path = confusion_dir / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix(y_true, y_score, matrix_path, threshold=args.threshold, title=f"{model_name} confusion matrix")
    logger.info("Saved confusion matrix to %s", matrix_path)

    comparison_path = result_dir / "model_comparison.csv"
    row = {"model": model_name, **metrics}
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
        comparison = comparison.drop_duplicates(subset=["model"], keep="last")
    else:
        comparison = pd.DataFrame([row])
    comparison.to_csv(comparison_path, index=False)
    logger.info("Updated model comparison table at %s", comparison_path)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained solar flare classifier.")
    parser.add_argument("--csv", default=str(config.LABELS_DIR / "test_labels.csv"))
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "efficientnet_b0_best.pth"))
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
    parser.add_argument("--result-dir", default=str(config.RESULT_DIR))
    parser.add_argument("--confusion-dir", default=str(config.CONFUSION_MATRIX_DIR))
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
