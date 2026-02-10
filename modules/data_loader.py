import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def load_raw_chat(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Laadt een WhatsApp chat-export (.txt).
    Elke regel wordt één rij in kolom 'raw'.
    """
    logger.info(f"Start laden van raw chat: {path}")

    with open(path, "r", encoding=encoding) as f:
        lines = [line.rstrip("\n") for line in f]

    df = pd.DataFrame({"raw": lines})

    logger.info(f"Raw chat geladen ({len(df)} regels)")
    return df

def load_clean_csv(path: Path) -> pd.DataFrame:
    """
    Laadt opgeschoonde data uit CSV.
    """
    logger.info(f"CSV laden: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])

    logger.info(f"CSV geladen ({len(df)} rijen)")
    return df

def load_clean_parquet(path: Path) -> pd.DataFrame:
    """
    Laadt opgeschoonde data uit Parquet.
    """
    logger.info(f"Parquet laden: {path}")

    df = pd.read_parquet(path)

    logger.info(f"Parquet geladen ({len(df)} rijen)")
    return df