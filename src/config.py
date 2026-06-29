"""Project-wide defaults for the solar flare forecasting pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SDOBENCHMARK_DIR = RAW_DATA_DIR / "sdobenchmark"
ACTIVE_REGION_MAGNETOGRAMS_DIR = RAW_DATA_DIR / "active_region_magnetograms"
SURYABENCH_DIR = RAW_DATA_DIR / "suryabench"

LABELS_DIR = PROCESSED_DATA_DIR / "labels"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrices"
GRADCAM_DIR = OUTPUT_DIR / "gradcam_results"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
EVALUATION_DIR = OUTPUT_DIR / "evaluations"
SCRIPT_LOG_DIR = LOG_DIR / "scripts"
PIPELINE_RUN_DIR = OUTPUT_DIR / "pipeline_runs"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 7
SEED = 42

FLARE_THRESHOLD = 1e-5
PREDICTION_THRESHOLD = 0.5
DEVICE = "auto"

LABEL_COLUMN = "label"
IMAGE_PATH_COLUMN = "image_path"

IMAGE_COLUMN_CANDIDATES = (
    "image_path",
    "filepath",
    "file_path",
    "path",
    "filename",
    "file_name",
    "image",
    "image_name",
    "hmi_path",
    "magnetogram_path",
)

PEAK_FLUX_COLUMN_CANDIDATES = (
    "peak_flux",
    "future_peak_flux",
    "goes_peak_flux",
    "flare_peak_flux",
    "max_flux",
    "flux",
)

MODEL_NAMES = (
    "custom_cnn",
    "resnet18",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)


def project_dirs() -> tuple[Path, ...]:
    """Return every directory that should exist before running the pipeline."""
    return (
        RAW_DATA_DIR / "sdobenchmark",
        RAW_DATA_DIR / "active_region_magnetograms",
        RAW_DATA_DIR / "suryabench",
        PROCESSED_DATA_DIR / "sdobenchmark_hmi",
        PROCESSED_DATA_DIR / "sdobenchmark_aia",
        LABELS_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        SCRIPT_LOG_DIR,
        FIGURE_DIR,
        RESULT_DIR,
        CONFUSION_MATRIX_DIR,
        GRADCAM_DIR,
        EXPERIMENT_DIR,
        EVALUATION_DIR,
        PIPELINE_RUN_DIR,
    )


def ensure_project_dirs() -> None:
    """Create the standard project directories."""
    for directory in project_dirs():
        directory.mkdir(parents=True, exist_ok=True)
