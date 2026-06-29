from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.preprocessing import prepare_label_csvs, prepare_official_sdobenchmark_label_csvs
from src.utils import get_logger, script_log_path, setup_logging

logger = get_logger(__name__)


def main() -> None:
    config.ensure_project_dirs()
    setup_logging(log_file=script_log_path(__file__))
    parser = argparse.ArgumentParser(description="Prepare train/val/test binary flare label CSV files.")
    parser.add_argument("--metadata", required=True, help="Path to the original metadata CSV.")
    parser.add_argument("--test-metadata", default=None, help="Optional official test metadata CSV.")
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--peak-flux-col", default=None)
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--threshold", type=float, default=config.FLARE_THRESHOLD)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--output-dir", default=str(config.LABELS_DIR))
    parser.add_argument("--force", action="store_true", help="Recreate label CSV files if they already exist.")
    args = parser.parse_args()

    if args.test_metadata:
        paths = prepare_official_sdobenchmark_label_csvs(
            train_metadata_csv=args.metadata,
            test_metadata_csv=args.test_metadata,
            image_root=args.image_root,
            output_dir=args.output_dir,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
            threshold=args.threshold,
            val_size=args.val_size,
            overwrite=args.force,
        )
    else:
        paths = prepare_label_csvs(
            metadata_csv=args.metadata,
            output_dir=args.output_dir,
            image_root=args.image_root,
            image_col=args.image_col,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
            threshold=args.threshold,
            val_size=args.val_size,
            test_size=args.test_size,
            split_col=args.split_col,
            overwrite=args.force,
        )
    logger.info("Prepared label files:")
    for split_name, path in paths.items():
        logger.info("%s: %s", split_name, path)


if __name__ == "__main__":
    main()
