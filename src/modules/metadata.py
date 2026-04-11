"""Metadata detection and persistence helpers for processed chat datasets."""

import logging
import re
import json
from pathlib import Path
import pandas as pd
from src import config

logger = logging.getLogger(__name__)

WHATSAPP_CONTROL_CHARS = ["\u200e", "\u202a", "\u202c"]


def contains_control_char(text: str) -> bool:
    """Check whether text contains invisible control characters.

    :param text: Input text to inspect.
    :type text: str
    :return: ``True`` if any configured control character is present.
    :rtype: bool
    """
    return any(char in text for char in WHATSAPP_CONTROL_CHARS)


def has_real_text(text: str) -> bool:
    """Check whether text contains alphanumeric content.

    :param text: Input text to inspect.
    :type text: str
    :return: ``True`` when at least one alphanumeric character is present.
    :rtype: bool
    """
    return bool(re.search(r"[A-Za-z0-9]", text))


def detect_group_name(df: pd.DataFrame) -> str | None:
    """Detect the probable WhatsApp group name from the first sender value.

    :param df: Chat dataframe with a ``sender`` column.
    :type df: pd.DataFrame
    :return: Detected group name, or ``None`` when unavailable.
    :rtype: str | None
    """
    if df.empty or "sender" not in df.columns:
        return None
    return str(df.iloc[0]["sender"])


def is_system_message(row: pd.Series, group_name: str | None) -> bool:
    """Identify system messages using known WhatsApp patterns.

    :param row: Row containing message-level values.
    :type row: pd.Series
    :param group_name: Detected group name used as system-message hint.
    :type group_name: str | None
    :return: ``True`` when the row is considered a system message.
    :rtype: bool
    """
    sender = str(row.get("sender", ""))
    message = str(row.get("message", ""))

    if contains_control_char(message):
        return True

    if not has_real_text(message):
        return True

    if group_name and sender == group_name and len(message.split()) < 5:
        return True

    return False


def detect_chat_start(df: pd.DataFrame) -> pd.Timestamp | None:
    """Detect the first non-system message timestamp in the chat.

    :param df: Chat dataframe with ``sender``, ``message``, and ``datetime``.
    :type df: pd.DataFrame
    :return: Earliest non-system timestamp, or ``None`` if unavailable.
    :rtype: pd.Timestamp | None
    """
    required_columns = {"sender", "message", "datetime"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required columns for metadata generation")

    if df.empty:
        return None

    group_name = detect_group_name(df)

    df_copy = df.copy()
    df_copy["is_system"] = df_copy.apply(
        lambda row: is_system_message(row, group_name),
        axis=1,
    )

    df_filtered = df_copy[~df_copy["is_system"]]

    if df_filtered.empty:
        return None

    return df_filtered["datetime"].min()


def generate_metadata(df: pd.DataFrame, raw_file_path: Path) -> dict:
    """Generate metadata for a chat dataset and its raw file.

    :param df: Parsed chat dataframe.
    :type df: pd.DataFrame
    :param raw_file_path: Path to the source raw chat file.
    :type raw_file_path: Path
    :return: Metadata dictionary for downstream persistence.
    :rtype: dict
    """
    raw_start = df["datetime"].min() if not df.empty else None
    real_start = detect_chat_start(df)

    metadata = {
        "file_name": raw_file_path.name,
        "total_messages": int(len(df)),
        "raw_start_date": raw_start.isoformat() if raw_start else None,
        "real_start_date": real_start.isoformat() if real_start else None,
    }

    logger.info("Metadata generated")
    return metadata


def save_metadata(metadata: dict) -> None:
    """Persist metadata to the processed metadata JSON file.

    :param metadata: Metadata payload to serialize.
    :type metadata: dict
    :return: None.
    :rtype: None
    """
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"Metadata saved: {config.METADATA_FILE}")
