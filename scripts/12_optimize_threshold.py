from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import SolarFlareImageDataset, build_transforms, make_dataloader
from src.metrics import binary_classification_metrics
from src.models import get_model
from src.utils import ensure_dir, get_device, get_logger, save_json, setup_logging

logger = get_logger(__name__)


def _json_ready(row: pd.Series) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key, value in row.items():
        if pd.isna(value):
            output[key] = float("nan")
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        else:
            output[key] = float(value)
    return output


def _prediction_scores(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model_name", "efficientnet_b0")
    image_size = args.image_size or checkpoint.get("image_size", config.IMAGE_SIZE)

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Threshold tuning CSV: %s", args.csv)
    logger.info("Model: %s", model_name)

    dataset = SolarFlareImageDataset(
        args.csv,
        image_root=args.image_root,
        transform=build_transforms(image_size, train=False),
        image_col=args.image_col,
        path_col=args.path_col,
        label_col=args.label_col,
        channel=args.channel,
    )
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("Loaded threshold tuning dataset with %d samples", len(dataset))

    model = get_model(model_name, pretrained=False).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="score"):
            images = images.to(device, non_blocking=True)
            logits = model(images).view(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            y_score.extend(scores)
            y_true.extend(labels.numpy().astype(int).tolist())
    return y_true, y_score


def optimize_threshold(args: argparse.Namespace) -> dict[str, float | int]:
    setup_logging()
    logger.info("Starting threshold optimization")
    y_true, y_score = _prediction_scores(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model or checkpoint.get("model_name", "efficientnet_b0")

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.steps)
    rows = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_score, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metrics})

    sweep = pd.DataFrame(rows)
    sort_columns = []
    for column in (args.metric, "tss", "hss", "f1", "recall", "precision"):
        if column in sweep.columns and column not in sort_columns:
            sort_columns.append(column)
    best = sweep.sort_values(sort_columns, ascending=[False] * len(sort_columns)).iloc[0]

    result_dir = ensure_dir(args.result_dir)
    split_name = Path(args.csv).stem.replace("_labels", "")
    sweep_path = result_dir / f"{model_name}_{split_name}_threshold_sweep.csv"
    best_path = result_dir / f"{model_name}_{split_name}_best_threshold_{args.metric}.json"

    sweep.to_csv(sweep_path, index=False)
    best_data = _json_ready(best)
    best_data["optimized_metric"] = args.metric
    best_data["source_csv"] = str(args.csv)
    best_data["checkpoint"] = str(args.checkpoint)
    save_json(best_data, best_path)

    logger.info("Saved threshold sweep to %s", sweep_path)
    logger.info("Saved best threshold to %s", best_path)
    logger.info(
        "Best threshold for %s: %.4f | tss=%.4f hss=%.4f f1=%.4f recall=%.4f precision=%.4f",
        args.metric,
        best_data["threshold"],
        best_data["tss"],
        best_data["hss"],
        best_data["f1"],
        best_data["recall"],
        best_data["precision"],
    )
    return best_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find a validation threshold for better TSS/HSS/F1.")
    parser.add_argument("--csv", default=str(config.LABELS_DIR / "val_labels.csv"))
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
    parser.add_argument("--metric", choices=("tss", "hss", "f1", "precision", "recall"), default="tss")
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--result-dir", default=str(config.RESULT_DIR))
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    optimize_threshold(args)


if __name__ == "__main__":
    main()
