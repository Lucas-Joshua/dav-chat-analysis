"""Application entry point for running the chat analysis pipeline."""

import argparse
import logging

from src import config
from src.logging_config import setup_logging
from src.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """Build a small CLI for selecting the raw chat input file."""
    parser = argparse.ArgumentParser(description="Run the DAV WhatsApp chat pipeline.")
    parser.add_argument(
        "raw_path",
        nargs="?",
        default=str(config.RAW_DATA_FILE),
        help="Path to the WhatsApp export text file.",
    )
    return parser


def main() -> None:
    """Run the data processing pipeline on the default raw dataset.

    :return: None.
    :rtype: None
    """
    args = _build_parser().parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DAV chat pipeline...")
    run_pipeline(args.raw_path)
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
