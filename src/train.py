"""Training entry point for image classification baselines."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from . import config
from .dataset import SolarFlareImageDataset, build_transforms, make_dataloader
from .metrics import binary_classification_metrics
from .models import FocalLoss, get_model
from .utils import ensure_dir, get_device, save_csv_rows, set_seed


def compute_pos_weight(labels: pd.Series) -> torch.Tensor:
    labels = labels.astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


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
    set_seed(args.seed)
    config.ensure_project_dirs()

    device = get_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

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

    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.model, pretrained=args.pretrained).to(device)
    if args.loss == "focal":
        criterion = FocalLoss()
    else:
        pos_weight = compute_pos_weight(train_dataset.frame[args.label_col]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    if use_amp and hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    elif use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    log_dir = ensure_dir(args.log_dir)
    checkpoint_path = checkpoint_dir / f"{args.model}_best.pth"
    log_path = log_dir / f"{args.model}_training_log.csv"

    rows: list[dict[str, float | int]] = []
    best_score = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
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
        save_csv_rows(rows, log_path)

        monitor = metrics["pr_auc"]
        if pd.isna(monitor):
            monitor = metrics["f1"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"f1={metrics['f1']:.4f} pr_auc={metrics['pr_auc']:.4f} tss={metrics['tss']:.4f}"
        )

        if float(monitor) > best_score:
            best_score = float(monitor)
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
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Best checkpoint saved to {checkpoint_path}")
    print(f"Training log saved to {log_path}")
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
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--log-dir", default=str(config.LOG_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
