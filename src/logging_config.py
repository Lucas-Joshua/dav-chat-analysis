import logging
from pathlib import Path
import colorlog
from src import config


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_to_file: bool = False) -> None:
    """
    Configure root logger with colored console output.
    Optionally enable file logging to logs directory.
    """

    root_logger = logging.getLogger()

    # Prevent duplicate handlers if called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    # ----------------------------
    # Console handler (colored)
    # ----------------------------
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

    # ----------------------------
    # Optional file logging
    # ----------------------------
    if log_to_file:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.LOGS_DIR / f"pipeline_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)