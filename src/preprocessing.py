"""Metadata cleaning, label preparation, and dataset exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config
from .utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def label_output_paths(output_dir: str | Path = config.LABELS_DIR) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "train": output_dir / "train_labels.csv",
        "val": output_dir / "val_labels.csv",
        "test": output_dir / "test_labels.csv",
    }


def prepared_label_files_exist(output_dir: str | Path = config.LABELS_DIR) -> bool:
    return all(path.exists() for path in label_output_paths(output_dir).values())


def existing_label_files(output_dir: str | Path = config.LABELS_DIR) -> list[Path]:
    return [path for path in label_output_paths(output_dir).values() if path.exists()]


def convert_flux_to_label(peak_flux: float, threshold: float = config.FLARE_THRESHOLD) -> int:
    """Convert peak flux into a binary M-class-or-stronger label."""
    try:
        return int(float(peak_flux) >= threshold)
    except (TypeError, ValueError):
        return 0


def read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    contains: str | None = None,
) -> str | None:
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    if contains:
        needle = contains.lower()
        for column in frame.columns:
            if needle in column.lower():
                return column
    return None


def find_peak_flux_column(frame: pd.DataFrame) -> str:
    column = find_column(frame, config.PEAK_FLUX_COLUMN_CANDIDATES, contains="flux")
    if column is None:
        raise ValueError(
            "Could not find a peak-flux column. Pass --peak-flux-col or rename the "
            "metadata column to peak_flux."
        )
    return column


def find_image_column(frame: pd.DataFrame, channel: str | None = None) -> str | None:
    if channel:
        channel_lower = channel.lower()
        path_columns = [
            column
            for column in frame.columns
            if channel_lower in column.lower()
            and any(token in column.lower() for token in ("path", "file", "image"))
        ]
        if path_columns:
            return path_columns[0]
    return find_column(frame, config.IMAGE_COLUMN_CANDIDATES)


def channel_to_file_token(channel: str | None) -> str | None:
    if channel is None:
        return None
    channel = channel.lower().strip()
    aliases = {
        "hmi": "magnetogram",
        "mag": "magnetogram",
        "magnetogram": "magnetogram",
        "continuum": "continuum",
        "aia94": "94",
        "aia131": "131",
        "aia171": "171",
        "aia193": "193",
        "aia211": "211",
        "aia304": "304",
        "aia335": "335",
        "aia1700": "1700",
    }
    if channel in aliases:
        return aliases[channel]
    return channel.replace("aia_", "").replace("aia-", "").replace("aia", "")


def add_binary_labels(
    frame: pd.DataFrame,
    peak_flux_col: str | None = None,
    label_col: str = config.LABEL_COLUMN,
    threshold: float = config.FLARE_THRESHOLD,
) -> pd.DataFrame:
    frame = frame.copy()
    if label_col in frame.columns:
        frame[label_col] = frame[label_col].astype(int)
        return frame
    peak_flux_col = peak_flux_col or find_peak_flux_column(frame)
    frame[label_col] = frame[peak_flux_col].apply(lambda value: convert_flux_to_label(value, threshold))
    return frame


def discover_image_files(image_root: str | Path, channel: str | None = None) -> list[Path]:
    image_root = Path(image_root)
    if not image_root.exists():
        return []
    channel_token = channel_to_file_token(channel)
    files = [
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if channel_token:
        suffix = f"__{channel_token.lower()}"
        files = [path for path in files if path.stem.lower().endswith(suffix)]
    return files


def _resolve_path(value: object, image_root: str | Path | None) -> str:
    if pd.isna(value):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    candidates = []
    if image_root:
        candidates.append(Path(image_root) / path)
    candidates.append(config.PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else path)


def _row_tokens(row: pd.Series) -> list[str]:
    token_columns = (
        "id",
        "sample_id",
        "record_id",
        "harpnum",
        "harp",
        "noaa_ar",
        "ar",
        "active_region",
        "filename",
        "file_name",
    )
    tokens: list[str] = []
    for column in row.index:
        if column.lower() in token_columns:
            value = row[column]
            if not pd.isna(value):
                token = str(value).strip()
                if token:
                    tokens.append(token.lower())
    return tokens


def _sdobenchmark_sample_parts(sample_id: object) -> tuple[str, str] | None:
    if pd.isna(sample_id):
        return None
    value = str(sample_id).strip()
    if not value or "_" not in value:
        return None
    active_region, sample_folder = value.split("_", 1)
    if not active_region or not sample_folder:
        return None
    return active_region, sample_folder


def _latest_sdobenchmark_image(
    row: pd.Series,
    image_root: str | Path,
    channel: str | None = "hmi",
) -> str:
    if "id" not in row.index:
        return ""

    parts = _sdobenchmark_sample_parts(row["id"])
    if parts is None:
        return ""

    active_region, sample_folder = parts
    root = Path(image_root)
    candidate_roots = [
        root,
        root / "training",
        root / "test",
    ]
    channel_token = channel_to_file_token(channel)
    pattern = f"*__{channel_token}.jpg" if channel_token else "*.jpg"

    for candidate_root in candidate_roots:
        sample_dir = candidate_root / active_region / sample_folder
        if not sample_dir.exists():
            continue
        matches = sorted(sample_dir.glob(pattern))
        if matches:
            return str(matches[-1].resolve())
    return ""


def attach_image_paths(
    frame: pd.DataFrame,
    image_root: str | Path | None = None,
    image_col: str | None = None,
    channel: str | None = "hmi",
    output_col: str = config.IMAGE_PATH_COLUMN,
) -> pd.DataFrame:
    """Add a normalized image_path column from metadata or filename lookup."""
    frame = frame.copy()
    image_col = image_col or find_image_column(frame, channel=channel)

    if image_col:
        frame[output_col] = frame[image_col].apply(lambda value: _resolve_path(value, image_root))
        return frame

    if not image_root:
        return frame

    if "id" in frame.columns:
        resolved = [
            _latest_sdobenchmark_image(row, image_root=image_root, channel=channel)
            for _, row in frame.iterrows()
        ]
        if any(resolved):
            frame[output_col] = resolved
            return frame

    files = discover_image_files(image_root, channel=channel)
    lower_names = [(path.as_posix().lower(), path) for path in files]

    resolved: list[str] = []
    for _, row in frame.iterrows():
        tokens = _row_tokens(row)
        match = ""
        for token in tokens:
            found = next((path for name, path in lower_names if token in name), None)
            if found:
                match = str(found)
                break
        resolved.append(match)
    frame[output_col] = resolved
    return frame


def prepare_official_sdobenchmark_label_csvs(
    train_metadata_csv: str | Path,
    test_metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path = config.LABELS_DIR,
    peak_flux_col: str | None = None,
    label_col: str = config.LABEL_COLUMN,
    channel: str | None = "hmi",
    threshold: float = config.FLARE_THRESHOLD,
    val_size: float = 0.15,
    seed: int = config.SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Prepare labels from SDOBenchmark's official training/test split."""
    output_paths = label_output_paths(output_dir)
    existing_paths = existing_label_files(output_dir)
    if existing_paths and not overwrite:
        if len(existing_paths) == len(output_paths):
            logger.info("Prepared label files already exist in %s; skipping preprocessing.", output_dir)
            logger.info("Use --force to recreate them.")
            return output_paths
        examples = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Some prepared label files already exist and will not be overwritten: "
            f"{examples}. Use --force to recreate the full split set."
        )

    logger.info("Loading metadata from %s and %s", train_metadata_csv, test_metadata_csv)
    train_frame = read_metadata(train_metadata_csv)
    test_frame = read_metadata(test_metadata_csv)

    train_frame = add_binary_labels(train_frame, peak_flux_col=peak_flux_col, label_col=label_col, threshold=threshold)
    test_frame = add_binary_labels(test_frame, peak_flux_col=peak_flux_col, label_col=label_col, threshold=threshold)

    image_root = Path(image_root)
    train_frame = attach_image_paths(train_frame, image_root=image_root / "training", channel=channel)
    test_frame = attach_image_paths(test_frame, image_root=image_root / "test", channel=channel)

    train_frame = train_frame[train_frame[config.IMAGE_PATH_COLUMN].astype(str).str.len() > 0].copy()
    test_frame = test_frame[test_frame[config.IMAGE_PATH_COLUMN].astype(str).str.len() > 0].copy()
    logger.info("Resolved image paths for %d training rows and %d test rows", len(train_frame), len(test_frame))

    stratify = train_frame[label_col] if train_frame[label_col].nunique() > 1 else None
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None

    train_split, val_split = train_test_split(
        train_frame,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )

    output_dir = ensure_dir(output_dir)
    output_paths = label_output_paths(output_dir)
    train_split.to_csv(output_paths["train"], index=False)
    val_split.to_csv(output_paths["val"], index=False)
    test_frame.to_csv(output_paths["test"], index=False)
    logger.info("Saved train labels to %s", output_paths["train"])
    logger.info("Saved val labels to %s", output_paths["val"])
    logger.info("Saved test labels to %s", output_paths["test"])
    return output_paths


def check_images(
    frame: pd.DataFrame,
    path_col: str = config.IMAGE_PATH_COLUMN,
    limit: int | None = None,
) -> dict[str, list[str] | int]:
    missing: list[str] = []
    corrupt: list[str] = []
    checked = 0

    if path_col not in frame.columns:
        return {"checked": 0, "missing": missing, "corrupt": corrupt}

    for path_value in frame[path_col].dropna().astype(str):
        if limit is not None and checked >= limit:
            break
        checked += 1
        path = Path(path_value)
        if not path.exists():
            missing.append(path_value)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(path_value)

    return {"checked": checked, "missing": missing, "corrupt": corrupt}


def split_dataframe(
    frame: pd.DataFrame,
    label_col: str = config.LABEL_COLUMN,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = config.SEED,
    split_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    if split_col and split_col in frame.columns:
        splits = {name: part.copy() for name, part in frame.groupby(frame[split_col].astype(str).str.lower())}
        return {
            "train": splits.get("train", pd.DataFrame(columns=frame.columns)),
            "val": splits.get("val", splits.get("valid", splits.get("validation", pd.DataFrame(columns=frame.columns)))),
            "test": splits.get("test", pd.DataFrame(columns=frame.columns)),
        }

    stratify = frame[label_col] if label_col in frame.columns and frame[label_col].nunique() > 1 else None
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None

    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_val_stratify = train_val[label_col] if stratify is not None and train_val[label_col].nunique() > 1 else None
    if train_val_stratify is not None and train_val_stratify.value_counts().min() < 2:
        train_val_stratify = None

    val_fraction = val_size / max(1e-9, 1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_stratify,
    )
    return {"train": train.copy(), "val": val.copy(), "test": test.copy()}


def save_class_distribution_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    label_col: str = config.LABEL_COLUMN,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    counts = frame[label_col].value_counts().sort_index()
    labels = ["No flare", "Flare"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color=["#4f6f8f", "#d95f59"])
    plt.ylabel("Samples")
    plt.title("Binary flare label distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def prepare_label_csvs(
    metadata_csv: str | Path,
    output_dir: str | Path = config.LABELS_DIR,
    image_root: str | Path | None = config.SDOBENCHMARK_DIR,
    image_col: str | None = None,
    peak_flux_col: str | None = None,
    label_col: str = config.LABEL_COLUMN,
    channel: str | None = "hmi",
    threshold: float = config.FLARE_THRESHOLD,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = config.SEED,
    split_col: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_paths = label_output_paths(output_dir)
    existing_paths = existing_label_files(output_dir)
    if existing_paths and not overwrite:
        if len(existing_paths) == len(output_paths):
            logger.info("Prepared label files already exist in %s; skipping preprocessing.", output_dir)
            logger.info("Use --force to recreate them.")
            return output_paths
        examples = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Some prepared label files already exist and will not be overwritten: "
            f"{examples}. Use --force to recreate the full split set."
        )

    frame = read_metadata(metadata_csv)
    frame = add_binary_labels(frame, peak_flux_col=peak_flux_col, label_col=label_col, threshold=threshold)
    frame = attach_image_paths(frame, image_root=image_root, image_col=image_col, channel=channel)
    if config.IMAGE_PATH_COLUMN in frame.columns:
        frame = frame[frame[config.IMAGE_PATH_COLUMN].astype(str).str.len() > 0].copy()

    splits = split_dataframe(
        frame,
        label_col=label_col,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        split_col=split_col,
    )

    output_dir = ensure_dir(output_dir)
    output_paths = label_output_paths(output_dir)
    for split_name, split_frame in splits.items():
        split_frame.to_csv(output_paths[split_name], index=False)
    return output_paths


def explore_dataset(
    metadata_csv: str | Path,
    image_root: str | Path | None = config.SDOBENCHMARK_DIR,
    output_dir: str | Path = config.FIGURE_DIR,
    image_col: str | None = None,
    peak_flux_col: str | None = None,
    channel: str | None = "hmi",
    label_col: str = config.LABEL_COLUMN,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    summary_path = output_dir / "dataset_exploration.json"
    plot_path = output_dir / "class_distribution.png"
    if summary_path.exists() and plot_path.exists() and not overwrite:
        logger.info("Dataset exploration outputs already exist in %s; skipping exploration.", output_dir)
        logger.info("Use --force to recreate them.")
        return load_json(summary_path)

    logger.info("Exploring dataset from %s", metadata_csv)
    frame = read_metadata(metadata_csv)
    frame = add_binary_labels(frame, peak_flux_col=peak_flux_col, label_col=label_col)
    frame = attach_image_paths(frame, image_root=image_root, image_col=image_col, channel=channel)

    save_class_distribution_plot(frame, plot_path, label_col=label_col)
    image_report = check_images(frame)

    class_counts = frame[label_col].value_counts().sort_index().to_dict()
    summary = {
        "metadata_csv": str(metadata_csv),
        "num_samples": int(len(frame)),
        "num_columns": int(len(frame.columns)),
        "columns": list(frame.columns),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "image_report": {
            "checked": image_report["checked"],
            "missing_count": len(image_report["missing"]),
            "corrupt_count": len(image_report["corrupt"]),
            "missing_examples": image_report["missing"][:10],
            "corrupt_examples": image_report["corrupt"][:10],
        },
        "class_distribution_plot": str(plot_path),
    }
    save_json(summary, summary_path)
    logger.info("Saved dataset exploration summary to %s", summary_path)
    logger.info("Saved class distribution plot to %s", plot_path)
    return summary
