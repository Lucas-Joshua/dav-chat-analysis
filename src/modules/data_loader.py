"""I/O helpers for loading raw and cleaned chat datasets."""

import logging
from pathlib import Path

import pandas as pd
from src import config

logger = logging.getLogger(__name__)


def load_raw_chat(path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Load raw WhatsApp export lines into a dataframe with a ``raw`` column.

    :param path: Path to the raw chat export file.
    :type path: Path
    :param encoding: Optional file encoding override.
    :type encoding: str | None
    :return: Dataframe containing one row per raw line.
    :rtype: pd.DataFrame
    """
    if encoding is None:
        encoding = getattr(config, "ENCODING", "utf-8")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading raw chat: {path}")

    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    df = pd.DataFrame({"raw": lines})
    logger.info(f"Raw chat loaded ({len(df)} rows)")
    return df


def load_clean_csv(path: Path) -> pd.DataFrame:
    """Load a cleaned CSV dataset and parse datetime columns.

    :param path: Path to the cleaned CSV file.
    :type path: Path
    :return: Loaded dataframe with parsed ``datetime`` where available.
    :rtype: pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    logger.info(f"CSV loaded ({len(df)} rows)")
    return df


def load_clean_parquet(path: Path) -> pd.DataFrame:
    """Load a cleaned Parquet dataset.

    :param path: Path to the cleaned parquet file.
    :type path: Path
    :return: Loaded dataframe.
    :rtype: pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Loading Parquet: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Parquet loaded ({len(df)} rows)")
    return df
