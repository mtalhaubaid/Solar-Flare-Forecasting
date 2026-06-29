"""Shared utility functions."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure console logging and, when requested, a timestamped log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    logger = logging.getLogger("solar_flare")
    if log_file is not None:
        logger.info("Saving run log to %s", Path(log_file))
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "solar_flare")


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

    logger = logging.getLogger("solar_flare")
    try:
        return torch.device(device_name)
    except RuntimeError:
        logger.warning("PyTorch does not recognize device '%s'. Falling back to CPU.", device_name)
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
    logger = logging.getLogger("solar_flare")

    preference = (preference or "auto").lower().strip()

    if preference in {"auto", "best", "accelerator"}:
        device = _best_available_device()
        logger.info("Using device: %s", device)
        return device

    if preference == "gpu":
        device = _best_available_device()
        if device.type == "cpu":
            logger.warning("No GPU/accelerator backend available. Using device: cpu")
        else:
            logger.info("Using device: %s", device)
        return device

    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            device = _torch_device_or_cpu(preference)
            logger.info("Using device: %s", device)
            return device
        logger.warning("CUDA was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference.startswith("xpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = _torch_device_or_cpu(preference)
            logger.info("Using device: %s", device)
            return device
        logger.warning("XPU was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using device: %s", device)
            return device
        logger.warning("MPS was requested but is not available. Using device: cpu")
        return torch.device("cpu")

    if preference in {"cpu", "vpu", "npu"}:
        if preference in {"vpu", "npu"}:
            logger.warning("%s is not configured as a PyTorch backend here. Using device: cpu", preference.upper())
        else:
            logger.info("Using device: cpu")
        return torch.device("cpu")

    device = _torch_device_or_cpu(preference)
    logger.info("Using device: %s", device)
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


def safe_filename_part(value: object) -> str:
    """Return a filesystem-friendly token for run names and artifact filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def build_run_name(
    model_name: str,
    epochs: int | str | None = None,
    run_type: str | None = None,
    created_at: str | None = None,
    suffix: str | None = None,
) -> str:
    """Build a timestamped run name with model and epoch information."""
    parts = [created_at or timestamp()]
    if run_type:
        parts.append(safe_filename_part(run_type))
    parts.append(safe_filename_part(model_name))
    if epochs is not None:
        parts.append(f"epochs{safe_filename_part(epochs)}")
    if suffix:
        parts.append(safe_filename_part(suffix))
    return "_".join(parts)


def script_log_path(script_file: str | Path, created_at: str | None = None) -> Path:
    """Return the default timestamped log path for a standalone script."""
    from . import config

    script_name = safe_filename_part(Path(script_file).stem)
    return config.SCRIPT_LOG_DIR / f"{created_at or timestamp()}_{script_name}.log"


def copy_file_if_exists(source: str | Path, destination: str | Path) -> Path | None:
    """Copy a file into an artifact folder when it exists."""
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return None
    destination_path = Path(destination)
    ensure_dir(destination_path.parent)
    shutil.copy2(source_path, destination_path)
    return destination_path
