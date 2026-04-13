"""I/O helpers for loading raw and cleaned chat datasets."""

import logging
from pathlib import Path

import pandas as pd
from src import config

logger = logging.getLogger(__name__)

FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "cp1252",
    "latin-1",
)


def load_raw_chat(path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Load raw WhatsApp export lines into a dataframe with a ``raw`` column.

    :param path: Path to the raw chat export file.
    :type path: Path
    :param encoding: Optional file encoding override.
    :type encoding: str | None
    :return: Dataframe containing one row per raw line.
    :rtype: pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading raw chat: {path}")

    encodings_to_try = (
        (encoding,) if encoding is not None else (getattr(config, "ENCODING", "utf-8"), *FALLBACK_ENCODINGS)
    )

    last_error: UnicodeError | None = None
    lines: list[str] | None = None
    used_encoding = None
    for candidate in dict.fromkeys(encodings_to_try):
        try:
            with open(path, "r", encoding=candidate) as f:
                lines = [line.rstrip("\n") for line in f]
            used_encoding = candidate
            break
        except UnicodeError as exc:
            last_error = exc

    if lines is None:
        if last_error is not None:
            raise last_error
        raise UnicodeError("Unable to decode raw chat file with supported encodings")

    df = pd.DataFrame({"raw": lines})
    logger.info("Raw chat loaded (%d rows, encoding=%s)", len(df), used_encoding)
    return df
