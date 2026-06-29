# python scripts/00_run_all.py --models all --skip-gradcam
# python scripts/00_run_all.py --models efficientnet_b0 convnext_tiny --epochs 30
# python scripts/00_run_all.py --skip-threshold --skip-gradcam --epochs 5



from __future__ import annotations

import argparse
import importlib.util
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.preprocessing import explore_dataset, prepare_label_csvs, prepare_official_sdobenchmark_label_csvs
from src.utils import ensure_dir, get_logger, save_json, script_log_path, setup_logging, timestamp

logger = get_logger(__name__)

DEFAULT_MODELS = ("resnet18", "efficientnet_b0", "convnext_tiny", "swin_t", "vit_b_16")


def _dataset_exists(dataset_dir: Path) -> bool:
    training_meta = dataset_dir / "training" / "meta_data.csv"
    test_meta = dataset_dir / "test" / "meta_data.csv"
    if training_meta.exists() and test_meta.exists():
        return True
    return dataset_dir.exists() and any(
        path.is_file() and path.suffix.lower() != ".zip" for path in dataset_dir.rglob("*")
    )


def _download_file(url: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, output_path)
    logger.info("Saved archive to %s", output_path)


def _extract_archive(output_path: Path) -> None:
    if output_path.suffix.lower() != ".zip":
        raise ValueError("--extract currently supports .zip archives only.")
    with zipfile.ZipFile(output_path, "r") as archive:
        archive.extractall(output_path.parent)
    logger.info("Extracted archive into %s", output_path.parent)


def _maybe_download_dataset(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.image_root)
    archive_path = Path(args.archive_output)
    if args.skip_download:
        logger.info("Skipping download step because --skip-download was provided.")
        return

    if _dataset_exists(dataset_dir) and not args.force_download:
        logger.info("Dataset already exists at %s; skipping download.", dataset_dir)
        return

    if archive_path.exists() and not args.force_download:
        logger.info("Dataset archive already exists at %s; skipping download.", archive_path)
        if args.extract:
            _extract_archive(archive_path)
        return

    if not args.dataset_url:
        logger.info("No dataset URL provided. Assuming dataset files are already available.")
        return

    _download_file(args.dataset_url, archive_path)
    if args.extract:
        _extract_archive(archive_path)


def _load_threshold_module() -> ModuleType:
    module_path = Path(__file__).with_name("09_optimize_threshold.py")
    spec = importlib.util.spec_from_file_location("solar_flare_threshold_optimizer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load threshold optimizer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restore_pipeline_logging(log_path: Path) -> None:
    setup_logging(log_file=log_path)


def _train_model(args: argparse.Namespace, model_name: str, pipeline_timestamp: str) -> Path:
    from src.train import build_arg_parser as build_train_parser
    from src.train import train

    train_args = build_train_parser().parse_args(
        [
            "--train-csv",
            str(args.train_csv),
            "--val-csv",
            str(args.val_csv),
            "--image-root",
            str(args.image_root),
            "--model",
            model_name,
            "--channel",
            args.channel,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--threshold",
            str(args.threshold),
            "--loss",
            args.loss,
            "--run-timestamp",
            pipeline_timestamp,
        ]
    )
    if args.pretrained:
        train_args.pretrained = True
    if args.no_amp:
        train_args.amp = False
    return train(train_args)


def _optimize_threshold(args: argparse.Namespace, checkpoint_path: Path, pipeline_timestamp: str) -> dict[str, float | int]:
    module = _load_threshold_module()
    threshold_args = module.build_arg_parser().parse_args(
        [
            "--csv",
            str(args.val_csv),
            "--image-root",
            str(args.image_root),
            "--checkpoint",
            str(checkpoint_path),
            "--channel",
            args.channel,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--metric",
            args.threshold_metric,
            "--steps",
            str(args.threshold_steps),
            "--run-timestamp",
            pipeline_timestamp,
        ]
    )
    return module.optimize_threshold(threshold_args)


def _evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    pipeline_timestamp: str,
    threshold: float,
) -> dict[str, float | int]:
    from src.evaluate import build_arg_parser as build_evaluate_parser
    from src.evaluate import evaluate

    evaluate_args = build_evaluate_parser().parse_args(
        [
            "--csv",
            str(args.test_csv),
            "--image-root",
            str(args.image_root),
            "--checkpoint",
            str(checkpoint_path),
            "--channel",
            args.channel,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--threshold",
            str(threshold),
            "--run-timestamp",
            pipeline_timestamp,
        ]
    )
    return evaluate(evaluate_args)


def _generate_gradcam(args: argparse.Namespace, checkpoint_path: Path, pipeline_timestamp: str, threshold: float) -> None:
    from src.gradcam import build_arg_parser as build_gradcam_parser
    from src.gradcam import generate_gradcam

    gradcam_args = build_gradcam_parser().parse_args(
        [
            "--csv",
            str(args.test_csv),
            "--image-root",
            str(args.image_root),
            "--checkpoint",
            str(checkpoint_path),
            "--channel",
            args.channel,
            "--device",
            args.device,
            "--threshold",
            str(threshold),
            "--limit",
            str(args.gradcam_limit),
            "--run-timestamp",
            pipeline_timestamp,
        ]
    )
    generate_gradcam(gradcam_args)


def run_all(args: argparse.Namespace) -> Path:
    config.ensure_project_dirs()
    if "all" in args.models:
        args.models = list(DEFAULT_MODELS)
    pipeline_timestamp = args.run_timestamp or timestamp()
    pipeline_name = f"{pipeline_timestamp}_run_all"
    pipeline_dir = ensure_dir(Path(args.pipeline_output_root) / pipeline_name)
    pipeline_log_path = script_log_path(__file__, created_at=pipeline_timestamp)

    setup_logging(log_file=pipeline_log_path)
    logger.info("Starting full solar flare pipeline")
    logger.info("Pipeline run directory: %s", pipeline_dir)
    logger.info("Models: %s", ", ".join(args.models))

    summary: dict[str, object] = {
        "pipeline_name": pipeline_name,
        "pipeline_timestamp": pipeline_timestamp,
        "pipeline_dir": str(pipeline_dir),
        "pipeline_log": str(pipeline_log_path),
        "models": args.models,
        "checkpoints": {},
        "thresholds": {},
        "evaluations": {},
        "gradcam": {},
    }

    _maybe_download_dataset(args)
    if not Path(args.metadata).exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    if not args.skip_explore:
        explore_dataset(
            metadata_csv=args.metadata,
            image_root=args.image_root,
            output_dir=args.explore_output_dir,
            image_col=args.image_col,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
            overwrite=args.force_explore,
        )
    else:
        logger.info("Skipping exploration step because --skip-explore was provided.")

    test_metadata = Path(args.test_metadata) if args.test_metadata else None
    if test_metadata is not None and not test_metadata.exists():
        logger.info("Official test metadata was not found at %s; using one metadata file split.", test_metadata)
        test_metadata = None

    if test_metadata is not None:
        label_paths = prepare_official_sdobenchmark_label_csvs(
            train_metadata_csv=args.metadata,
            test_metadata_csv=test_metadata,
            image_root=args.image_root,
            output_dir=args.label_output_dir,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
            threshold=args.flare_threshold,
            val_size=args.val_size,
            overwrite=args.force_preprocess,
        )
    else:
        label_paths = prepare_label_csvs(
            metadata_csv=args.metadata,
            output_dir=args.label_output_dir,
            image_root=args.image_root,
            image_col=args.image_col,
            peak_flux_col=args.peak_flux_col,
            channel=args.channel,
            threshold=args.flare_threshold,
            val_size=args.val_size,
            test_size=args.test_size,
            split_col=args.split_col,
            overwrite=args.force_preprocess,
        )

    args.train_csv = label_paths["train"]
    args.val_csv = label_paths["val"]
    args.test_csv = label_paths["test"]
    summary["label_paths"] = {key: str(value) for key, value in label_paths.items()}

    for model_name in args.models:
        _restore_pipeline_logging(pipeline_log_path)
        logger.info("Starting model workflow: %s", model_name)

        checkpoint_path = _train_model(args, model_name, pipeline_timestamp)
        summary["checkpoints"][model_name] = str(checkpoint_path)  # type: ignore[index]

        evaluation_threshold = args.threshold
        if not args.skip_threshold:
            threshold_result = _optimize_threshold(args, checkpoint_path, pipeline_timestamp)
            summary["thresholds"][model_name] = threshold_result  # type: ignore[index]
            if "threshold" in threshold_result:
                evaluation_threshold = float(threshold_result["threshold"])
        else:
            logger.info("Skipping threshold optimization for %s.", model_name)

        if not args.skip_evaluate:
            metrics = _evaluate_checkpoint(args, checkpoint_path, pipeline_timestamp, evaluation_threshold)
            summary["evaluations"][model_name] = metrics  # type: ignore[index]
        else:
            logger.info("Skipping evaluation for %s.", model_name)

        if not args.skip_gradcam:
            _generate_gradcam(args, checkpoint_path, pipeline_timestamp, evaluation_threshold)
            summary["gradcam"][model_name] = {  # type: ignore[index]
                "checkpoint": str(checkpoint_path),
                "threshold": evaluation_threshold,
                "limit": args.gradcam_limit,
            }
        else:
            logger.info("Skipping Grad-CAM for %s.", model_name)

        _restore_pipeline_logging(pipeline_log_path)
        logger.info("Finished model workflow: %s", model_name)
        save_json(summary, pipeline_dir / f"{pipeline_name}_summary.json")

    summary_path = pipeline_dir / f"{pipeline_name}_summary.json"
    save_json(summary, summary_path)
    logger.info("Full pipeline complete. Summary saved to %s", summary_path)
    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete solar flare experiment pipeline.")
    parser.add_argument("--dataset-url", default=None, help="Optional dataset archive URL.")
    parser.add_argument("--archive-output", default=str(config.SDOBENCHMARK_DIR / "sdobenchmark.zip"))
    parser.add_argument("--extract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--metadata", default=str(config.SDOBENCHMARK_DIR / "training" / "meta_data.csv"))
    parser.add_argument("--test-metadata", default=str(config.SDOBENCHMARK_DIR / "test" / "meta_data.csv"))
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--peak-flux-col", default=None)
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--explore-output-dir", default=str(config.FIGURE_DIR))
    parser.add_argument("--label-output-dir", default=str(config.LABELS_DIR))
    parser.add_argument("--skip-explore", action="store_true")
    parser.add_argument("--force-explore", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--flare-threshold", type=float, default=config.FLARE_THRESHOLD)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=(*config.MODEL_NAMES, "all"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    parser.add_argument("--threshold-metric", choices=("tss", "hss", "f1", "precision", "recall"), default="tss")
    parser.add_argument("--threshold-steps", type=int, default=501)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-threshold", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--gradcam-limit", type=int, default=16)
    parser.add_argument("--pipeline-output-root", default=str(config.PIPELINE_RUN_DIR))
    parser.add_argument("--run-timestamp", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
