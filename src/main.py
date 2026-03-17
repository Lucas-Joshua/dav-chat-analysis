"""Application entry point for running the chat analysis pipeline."""

import logging

from src.logging_config import setup_logging
from src.pipeline import run_pipeline


def main():
    """Run the data processing pipeline on the default raw dataset."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DAV chat pipeline...")
    run_pipeline("data/raw/raw_data.txt")
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
