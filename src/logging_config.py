"""Logging setup utilities for console and rotating file handlers."""

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import colorlog
from src import config


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level=logging.INFO, log_to_file=True):
    """Configure root logging for console output and optional log files.

    :param level: Logging level to apply to the root logger.
    :type level: int
    :param log_to_file: Whether to also write logs to a rotating file.
    :type log_to_file: bool
    :return: None.
    :rtype: None
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + LOG_FORMAT,
        datefmt=DATE_FORMAT,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.LOGS_DIR / f"pipeline_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )

        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)

    for noisy_logger in [
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "choreographer",
        "kaleido",
        "urllib3",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
