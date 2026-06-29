from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.utils import get_logger, script_log_path, setup_logging

logger = get_logger(__name__)


def main() -> None:
    config.ensure_project_dirs()
    setup_logging(log_file=script_log_path(__file__))
    logger.info(
        "Multi-channel training is reserved for the second milestone. "
        "Finish the HMI-only baseline first, then extend the dataset loader to "
        "return stacked HMI/AIA channels."
    )


if __name__ == "__main__":
    main()
