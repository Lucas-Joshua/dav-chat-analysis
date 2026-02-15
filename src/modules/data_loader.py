import logging
from pathlib import Path
import pandas as pd
from src import config

logger = logging.getLogger(__name__)


def load_raw_chat(path: Path, encoding: str | None = None) -> pd.DataFrame:
    if encoding is None:
        encoding = config.ENCODING

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading raw chat: {path}")

    with open(path, "r", encoding=encoding) as f:
        lines = [line.rstrip("\n") for line in f]

    df = pd.DataFrame({"raw": lines})

    logger.info(f"Raw chat loaded ({len(df)} rows)")
    return df


def load_clean_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info(f"Loading CSV: {path}")

    df = pd.read_csv(path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    logger.info(f"CSV loaded ({len(df)} rows)")
    return df


def load_clean_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Loading Parquet: {path}")

    df = pd.read_parquet(path)

    logger.info(f"Parquet loaded ({len(df)} rows)")
    return df