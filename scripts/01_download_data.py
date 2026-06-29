from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.utils import ensure_dir, get_logger, script_log_path, setup_logging

logger = get_logger(__name__)


def download_file(url: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, output_path)
    logger.info("Saved archive to %s", output_path)


def dataset_exists(dataset_dir: Path) -> bool:
    """Return True when the SDOBenchmark files we need are already present."""
    training_meta = dataset_dir / "training" / "meta_data.csv"
    test_meta = dataset_dir / "test" / "meta_data.csv"
    if training_meta.exists() and test_meta.exists():
        return True
    return dataset_dir.exists() and any(
        path.is_file() and path.suffix.lower() != ".zip" for path in dataset_dir.rglob("*")
    )


def extract_archive(output_path: Path) -> None:
    if output_path.suffix.lower() != ".zip":
        raise ValueError("--extract currently supports .zip archives only.")
    with zipfile.ZipFile(output_path, "r") as archive:
        archive.extractall(output_path.parent)
    logger.info("Extracted archive into %s", output_path.parent)


def main() -> None:
    config.ensure_project_dirs()
    setup_logging(log_file=script_log_path(__file__))
    parser = argparse.ArgumentParser(description="Download dataset archives into data/raw.")
    parser.add_argument("--url", default=None, help="Dataset archive URL.")
    parser.add_argument("--output", default=str(config.SDOBENCHMARK_DIR / "sdobenchmark.zip"))
    parser.add_argument("--extract", action="store_true", help="Extract zip archive after download.")
    parser.add_argument("--force", action="store_true", help="Download again even if the dataset already exists.")
    args = parser.parse_args()

    dataset_dir = Path(config.SDOBENCHMARK_DIR)
    output_path = Path(args.output)

    if dataset_exists(dataset_dir) and not args.force:
        logger.info("SDOBenchmark already exists at %s", dataset_dir)
        logger.info("Skipping download. Use --force if you want to download again.")
        return

    if output_path.exists() and not args.force:
        logger.info("Dataset archive already exists at %s", output_path)
        logger.info("Skipping download. Use --force if you want to download it again.")
        if args.extract:
            logger.info("Extracting existing archive instead of downloading again.")
            extract_archive(output_path)
        return

    if not args.url:
        logger.info("Project folders are ready.")
        logger.info("Pass --url when you have the SDOBenchmark download link, for example:")
        logger.info("python scripts/01_download_data.py --url <URL> --output %s --extract", output_path)
        return

    download_file(args.url, output_path)
    if args.extract:
        extract_archive(output_path)


if __name__ == "__main__":
    main()
