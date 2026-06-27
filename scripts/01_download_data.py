from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.utils import ensure_dir


def download_file(url: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def dataset_exists(dataset_dir: Path) -> bool:
    """Return True when the SDOBenchmark files we need are already present."""
    training_meta = dataset_dir / "training" / "meta_data.csv"
    test_meta = dataset_dir / "test" / "meta_data.csv"
    return training_meta.exists() and test_meta.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset archives into data/raw.")
    parser.add_argument("--url", default=None, help="Dataset archive URL.")
    parser.add_argument("--output", default=str(config.SDOBENCHMARK_DIR / "sdobenchmark.zip"))
    parser.add_argument("--extract", action="store_true", help="Extract zip archive after download.")
    parser.add_argument("--force", action="store_true", help="Download again even if the dataset already exists.")
    args = parser.parse_args()

    config.ensure_project_dirs()
    dataset_dir = Path(config.SDOBENCHMARK_DIR)
    output_path = Path(args.output)

    if dataset_exists(dataset_dir) and not args.force:
        print(f"SDOBenchmark already exists at {dataset_dir}")
        print("Skipping download. Use --force if you want to download again.")
        return

    if not args.url:
        print("Project folders are ready.")
        print("Pass --url when you have the SDOBenchmark download link, for example:")
        print(f"python scripts/01_download_data.py --url <URL> --output {output_path} --extract")
        return

    download_file(args.url, output_path)
    if args.extract:
        if output_path.suffix.lower() != ".zip":
            raise ValueError("--extract currently supports .zip archives only.")
        with zipfile.ZipFile(output_path, "r") as archive:
            archive.extractall(output_path.parent)
        print(f"Extracted archive into {output_path.parent}")


if __name__ == "__main__":
    main()
