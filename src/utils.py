"""Shared utility functions."""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def _torch_device_or_cpu(device_name: str):
    import torch

    try:
        return torch.device(device_name)
    except RuntimeError:
        print(f"PyTorch does not recognize device '{device_name}'. Falling back to CPU.")
        return torch.device("cpu")


def _best_available_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _torch_device_or_cpu("xpu")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device(preference: str = "auto"):
    """Select a compute device with safe fallback.

    Use ``auto`` for normal runs. It chooses CUDA GPU first, then Intel XPU,
    then Apple MPS, then CPU. Explicit unavailable devices fall back to CPU.
    """
    import torch

    preference = (preference or "auto").lower().strip()

    if preference in {"auto", "best", "accelerator"}:
        device = _best_available_device()
        print(f"Using device: {device}")
        return device

    if preference == "gpu":
        device = _best_available_device()
        if device.type == "cpu":
            print("No GPU/accelerator backend available. Using device: cpu")
        else:
            print(f"Using device: {device}")
        return device

    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            device = _torch_device_or_cpu(preference)
            print(f"Using device: {device}")
            return device
        print("CUDA was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference.startswith("xpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = _torch_device_or_cpu(preference)
            print(f"Using device: {device}")
            return device
        print("XPU was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using device: {device}")
            return device
        print("MPS was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference in {"cpu", "vpu", "npu"}:
        if preference in {"vpu", "npu"}:
            print(f"{preference.upper()} is not configured as a PyTorch backend here. Using device: cpu")
        else:
            print("Using device: cpu")
        return torch.device("cpu")

    device = _torch_device_or_cpu(preference)
    print(f"Using device: {device}")
    return device


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
