"""Evaluation metrics for binary flare prediction."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import ensure_dir

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

METRIC_DISPLAY_ORDER = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("tss", "TSS"),
    ("hss", "HSS"),
    ("tn", "True negatives"),
    ("fp", "False positives"),
    ("fn", "False negatives"),
    ("tp", "True positives"),
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def tss_score(tp: int, tn: int, fp: int, fn: int) -> float:
    tpr = _safe_divide(tp, tp + fn)
    fpr = _safe_divide(fp, fp + tn)
    return tpr - fpr


def hss_score(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = 2 * ((tp * tn) - (fn * fp))
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    return _safe_divide(numerator, denominator)


def binary_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tss": float(tss_score(tp, tn, fp, fn)),
        "hss": float(hss_score(tp, tn, fp, fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def _format_metric_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def format_binary_metrics(metrics: dict[str, float | int], title: str = "Metrics") -> str:
    """Return a readable multi-line metrics block for console logs."""
    lines = [title]
    for key, label in METRIC_DISPLAY_ORDER:
        if key in metrics:
            lines.append(f"  {label}: {_format_metric_value(metrics[key])}")
    return "\n".join(lines)


def format_compact_binary_metrics(metrics: dict[str, float | int]) -> str:
    """Return a single-line metrics summary for epoch logs."""
    parts = []
    for key, _ in METRIC_DISPLAY_ORDER:
        if key in metrics:
            parts.append(f"{key}={_format_metric_value(metrics[key])}")
    return " ".join(parts)


def log_binary_metrics(
    logger: logging.Logger,
    metrics: dict[str, float | int],
    title: str = "Metrics",
) -> None:
    """Log every metric line separately so console output stays timestamped."""
    for line in format_binary_metrics(metrics, title=title).splitlines():
        logger.info(line)


def save_confusion_matrix(
    y_true,
    y_score,
    output_path: str | Path,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
) -> None:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(5, 4))
    labels = ["No flare", "Flare"]
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        threshold_value = matrix.max() / 2.0 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                color = "white" if matrix[row_index, column_index] > threshold_value else "black"
                plt.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color=color,
                )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
