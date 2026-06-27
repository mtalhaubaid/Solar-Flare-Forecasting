from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.preprocessing import explore_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore SDOBenchmark metadata and HMI image availability.")
    parser.add_argument("--metadata", required=True, help="Path to the SDOBenchmark metadata CSV.")
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--peak-flux-col", default=None)
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--output-dir", default=str(config.FIGURE_DIR))
    args = parser.parse_args()

    summary = explore_dataset(
        metadata_csv=args.metadata,
        image_root=args.image_root,
        output_dir=args.output_dir,
        image_col=args.image_col,
        peak_flux_col=args.peak_flux_col,
        channel=args.channel,
    )
    print("Dataset exploration complete.")
    print(f"Samples: {summary['num_samples']}")
    print(f"Class counts: {summary['class_counts']}")
    print(f"Report saved in {args.output_dir}")


if __name__ == "__main__":
    main()

