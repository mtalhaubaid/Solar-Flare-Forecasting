"""Evaluation metrics for binary flare prediction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No flare", "Flare"],
        yticklabels=["No flare", "Flare"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

