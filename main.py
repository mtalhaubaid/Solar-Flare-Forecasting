from __future__ import annotations

import argparse

from src import config
from src.preprocessing import explore_dataset, prepare_label_csvs
from src.utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Solar flare forecasting research pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    explore_parser = subparsers.add_parser("explore", help="Explore metadata and label distribution.")
    explore_parser.add_argument("--metadata", required=True)
    explore_parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    explore_parser.add_argument("--image-col", default=None)
    explore_parser.add_argument("--peak-flux-col", default=None)
    explore_parser.add_argument("--channel", default="hmi")
    explore_parser.add_argument("--output-dir", default=str(config.FIGURE_DIR))

    labels_parser = subparsers.add_parser("prepare-labels", help="Create train/val/test label CSV files.")
    labels_parser.add_argument("--metadata", required=True)
    labels_parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    labels_parser.add_argument("--image-col", default=None)
    labels_parser.add_argument("--peak-flux-col", default=None)
    labels_parser.add_argument("--channel", default="hmi")
    labels_parser.add_argument("--threshold", type=float, default=config.FLARE_THRESHOLD)
    labels_parser.add_argument("--val-size", type=float, default=0.15)
    labels_parser.add_argument("--test-size", type=float, default=0.15)
    labels_parser.add_argument("--split-col", default=None)
    labels_parser.add_argument("--output-dir", default=str(config.LABELS_DIR))

    subparsers.add_parser("train", help="Train a model. Use scripts/05_train_efficientnet.py for full help.")
    subparsers.add_parser("evaluate", help="Evaluate a checkpoint. Use scripts/07_evaluate_model.py for full help.")
    subparsers.add_parser("gradcam", help="Generate Grad-CAM overlays. Use scripts/08_generate_gradcam.py for full help.")

    args, remaining = parser.parse_known_args()
    if args.command == "explore":
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        explore_dataset(
            metadata_csv=args.metadata,
            image_root=args.image_root,
            output_dir=args.output_dir,
            image_col=args.image_col,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
        )
    elif args.command == "prepare-labels":
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        prepare_label_csvs(
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
        )
    elif args.command == "train":
        from src.train import build_arg_parser, train

        train(build_arg_parser().parse_args(remaining))
    elif args.command == "evaluate":
        from src.evaluate import build_arg_parser, evaluate

        evaluate(build_arg_parser().parse_args(remaining))
    elif args.command == "gradcam":
        from src.gradcam import build_arg_parser, generate_gradcam

        generate_gradcam(build_arg_parser().parse_args(remaining))


if __name__ == "__main__":
    main()


# python scripts/01_download_data.py --url https://github.com/i4Ds/SDOBenchmark/archive/data-full.zip --output data/raw/sdobenchmark/sdobenchmark_full.zip --extract

# python scripts/02_explore_dataset.py --metadata data/raw/sdobenchmark/training/meta_data.csv --image-root data/raw/sdobenchmark --channel hmi

# python scripts/03_prepare_labels.py --metadata data/raw/sdobenchmark/training/meta_data.csv --test-metadata data/raw/sdobenchmark/test/meta_data.csv --image-root data/raw/sdobenchmark --channel hmi

# python scripts/03_prepare_labels.py --metadata data/raw/sdobenchmark/training/meta_data.csv --test-metadata data/raw/sdobenchmark/test/meta_data.csv --image-root data/raw/sdobenchmark --channel hmi

# python scripts/04_train_resnet18.py --device auto --epochs 30 --batch-size 16 --num-workers 0
# python scripts/05_train_efficientnet.py --device auto --epochs 30 --batch-size 16 --num-workers 0
# python scripts/06_train_convnext.py --device auto --epochs 30 --batch-size 8 --num-workers 0




# python scripts/07_evaluate_model.py --checkpoint outputs/checkpoints/resnet18_best.pth --model resnet18
# python scripts/07_evaluate_model.py --checkpoint outputs/checkpoints/efficientnet_b0_best.pth --model efficientnet_b0
# python scripts/07_evaluate_model.py --checkpoint outputs/checkpoints/convnext_tiny_best.pth --model convnext_tiny

# !python scripts/04_train_resnet18.py --device auto --epochs 30 --batch-size 16 --num-workers 2
# !python scripts/05_train_efficientnet.py --device auto --epochs 30 --batch-size 16 --num-workers 2
# !python scripts/06_train_convnext.py --device auto --epochs 30 --batch-size 8 --num-workers 2