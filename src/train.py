"""Training entry point for image classification baselines."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from . import config
from .dataset import SolarFlareImageDataset, build_transforms, make_dataloader
from .metrics import binary_classification_metrics, format_compact_binary_metrics, log_binary_metrics
from .models import FocalLoss, get_model
from .utils import (
    build_run_name,
    copy_file_if_exists,
    ensure_dir,
    get_device,
    get_logger,
    save_csv_rows,
    save_json,
    set_seed,
    setup_logging,
    timestamp,
)

logger = get_logger(__name__)


def compute_pos_weight(labels: pd.Series) -> torch.Tensor:
    labels = labels.astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


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


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _save_input_csvs(args: argparse.Namespace, data_dir: Path, run_name: str) -> dict[str, str]:
    copied: dict[str, str] = {}
    train_copy = copy_file_if_exists(args.train_csv, data_dir / f"{run_name}_train_labels.csv")
    val_copy = copy_file_if_exists(args.val_csv, data_dir / f"{run_name}_val_labels.csv")
    if train_copy is not None:
        copied["train_csv"] = str(train_copy)
    if val_copy is not None:
        copied["val_csv"] = str(val_copy)
    return copied


def save_training_curves(rows: list[dict[str, float | int]], output_path: str | Path) -> None:
    if not rows:
        return

    frame = pd.DataFrame(rows)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", label="train_loss")
    axes[0].plot(frame["epoch"], frame["val_loss"], marker="o", label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for column in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tss", "hss"):
        if column in frame.columns:
            axes[1].plot(frame["epoch"], frame[column], marker="o", label=column)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation metric")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_score: list[float] = []

    progress = tqdm(loader, leave=False, desc="train" if training else "valid")
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            amp_context = torch.amp.autocast("cuda") if use_amp and hasattr(torch, "amp") else nullcontext()
            with amp_context:
                logits = model(images).view(-1)
                loss = criterion(logits, labels)

            if training:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        scores = torch.sigmoid(logits.detach()).cpu().numpy().tolist()
        y_score.extend(scores)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        progress.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, len(loader.dataset)), y_true, y_score


def train(args: argparse.Namespace) -> Path:
    config.ensure_project_dirs()
    created_at = args.run_timestamp or timestamp()
    run_name = args.run_name or build_run_name(args.model, epochs=args.epochs, created_at=created_at)
    run_dir = ensure_dir(args.run_dir or (Path(args.output_root) / run_name))
    checkpoint_dir = ensure_dir(args.checkpoint_dir or (run_dir / "checkpoints"))
    log_dir = ensure_dir(args.log_dir or (run_dir / "logs"))
    result_dir = ensure_dir(run_dir / "results")
    figure_dir = ensure_dir(run_dir / "figures")
    data_dir = ensure_dir(run_dir / "data")
    log_path = log_dir / f"{run_name}_train.log"

    setup_logging(log_file=log_path)
    logger.info("Starting training for model=%s", args.model)
    logger.info("Training run name: %s", run_name)
    logger.info("Training run directory: %s", run_dir)
    set_seed(args.seed)

    device = get_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    logger.info("Using image root: %s", args.image_root)
    logger.info("Training CSV: %s", args.train_csv)
    logger.info("Validation CSV: %s", args.val_csv)
    logger.info(
        "Run settings: seed=%d epochs=%d image_size=%d batch_size=%d lr=%g weight_decay=%g "
        "threshold=%.4f patience=%d amp_requested=%s amp_enabled=%s",
        args.seed,
        args.epochs,
        args.image_size,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.threshold,
        args.patience,
        args.amp,
        use_amp,
    )
    copied_csvs = _save_input_csvs(args, data_dir, run_name)
    if copied_csvs:
        logger.info("Saved input CSV copies for this run: %s", copied_csvs)
    save_json(
        {
            "run_name": run_name,
            "run_type": "training",
            "created_at": created_at,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "result_dir": str(result_dir),
            "figure_dir": str(figure_dir),
            "data_dir": str(data_dir),
            "log_file": str(log_path),
            "input_csv_copies": copied_csvs,
            "args": _json_ready(vars(args)),
        },
        run_dir / f"{run_name}_run_config.json",
    )

    train_dataset = SolarFlareImageDataset(
        args.train_csv,
        image_root=args.image_root,
        transform=build_transforms(args.image_size, train=True),
        image_col=args.image_col,
        path_col=args.path_col,
        label_col=args.label_col,
        channel=args.channel,
    )
    val_dataset = SolarFlareImageDataset(
        args.val_csv,
        image_root=args.image_root,
        transform=build_transforms(args.image_size, train=False),
        image_col=args.image_col,
        path_col=args.path_col,
        label_col=args.label_col,
        channel=args.channel,
    )
    logger.info("Loaded training dataset with %d samples", len(train_dataset))
    logger.info("Loaded validation dataset with %d samples", len(val_dataset))
    logger.info("Training label distribution: %s", _label_distribution(train_dataset, args.label_col))
    logger.info("Validation label distribution: %s", _label_distribution(val_dataset, args.label_col))

    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Number of workers: %d", args.num_workers)
    logger.info("Training batches per epoch: %d", len(train_loader))
    logger.info("Validation batches per epoch: %d", len(val_loader))

    model = get_model(args.model, pretrained=args.pretrained).to(device)
    logger.info("Model initialized: %s", args.model)
    total_params, trainable_params = _parameter_counts(model)
    logger.info("Model parameters: total=%d trainable=%d", total_params, trainable_params)
    if args.loss == "focal":
        criterion = FocalLoss()
        logger.info("Using focal loss")
    else:
        pos_weight = compute_pos_weight(train_dataset.frame[args.label_col]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Using BCEWithLogitsLoss with pos_weight=%.4f", float(pos_weight.item()))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    if use_amp and hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    elif use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    checkpoint_path = checkpoint_dir / f"{run_name}_best.pth"
    final_checkpoint_path = checkpoint_dir / f"{run_name}_last.pth"
    training_log_path = log_dir / f"{run_name}_training_log.csv"
    training_curves_path = figure_dir / f"{run_name}_training_curves.png"
    logger.info("Checkpoints will be saved to %s", checkpoint_dir)
    logger.info("Training logs will be saved to %s", log_dir)
    logger.info("Training graphs will be saved to %s", figure_dir)

    rows: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch: int | None = None
    best_metrics: dict[str, float | int] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d started", epoch, args.epochs)
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, y_true, y_score = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        metrics = binary_classification_metrics(y_true, y_score, threshold=args.threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        save_csv_rows(rows, training_log_path)
        save_training_curves(rows, training_curves_path)

        monitor = metrics["pr_auc"]
        if pd.isna(monitor):
            monitor = metrics["f1"]

        logger.info(
            "Epoch %03d/%d completed | train_loss=%.4f val_loss=%.4f lr=%g | %s",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
            format_compact_binary_metrics(metrics),
        )

        if float(monitor) > best_score:
            best_score = float(monitor)
            best_epoch = epoch
            best_metrics = metrics.copy()
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "epoch": epoch,
                    "best_score": best_score,
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                    "label_col": args.label_col,
                    "path_col": args.path_col,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "created_at": created_at,
                    "epochs_requested": args.epochs,
                    "train_csv": str(args.train_csv),
                    "val_csv": str(args.val_csv),
                    "training_log": str(training_log_path),
                    "training_curves": str(training_curves_path),
                },
                checkpoint_path,
            )
            logger.info("New best checkpoint saved to %s (score=%.4f)", checkpoint_path, best_score)
        else:
            stale_epochs += 1
            logger.info("No validation improvement for %d/%d epoch(s)", stale_epochs, args.patience)
            if stale_epochs >= args.patience:
                logger.info("Early stopping after %d epochs.", epoch)
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "epoch": int(rows[-1]["epoch"]) if rows else 0,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "image_size": args.image_size,
            "threshold": args.threshold,
            "label_col": args.label_col,
            "path_col": args.path_col,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "created_at": created_at,
            "epochs_requested": args.epochs,
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "training_log": str(training_log_path),
            "training_curves": str(training_curves_path),
        },
        final_checkpoint_path,
    )

    summary_path = result_dir / f"{run_name}_training_summary.json"
    save_json(
        {
            "run_name": run_name,
            "run_type": "training",
            "created_at": created_at,
            "model_name": args.model,
            "epochs_requested": args.epochs,
            "epochs_completed": int(rows[-1]["epoch"]) if rows else 0,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_checkpoint": str(checkpoint_path),
            "last_checkpoint": str(final_checkpoint_path),
            "training_log": str(training_log_path),
            "training_curves": str(training_curves_path),
            "best_metrics": best_metrics or {},
            "run_dir": str(run_dir),
        },
        summary_path,
    )
    logger.info("Best checkpoint saved to %s", checkpoint_path)
    logger.info("Last checkpoint saved to %s", final_checkpoint_path)
    logger.info("Training log saved to %s", training_log_path)
    logger.info("Training curves saved to %s", training_curves_path)
    logger.info("Training summary saved to %s", summary_path)
    if best_metrics is not None and best_epoch is not None:
        log_binary_metrics(logger, best_metrics, title=f"Best validation metrics (epoch {best_epoch:03d})")
    return checkpoint_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a solar flare image classifier.")
    parser.add_argument("--train-csv", default=str(config.LABELS_DIR / "train_labels.csv"))
    parser.add_argument("--val-csv", default=str(config.LABELS_DIR / "val_labels.csv"))
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--model", default="efficientnet_b0", choices=config.MODEL_NAMES)
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--path-col", default=config.IMAGE_PATH_COLUMN)
    parser.add_argument("--label-col", default=config.LABEL_COLUMN)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default=str(config.EXPERIMENT_DIR))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-timestamp", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
