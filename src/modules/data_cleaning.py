"""Cleaning and parsing logic for WhatsApp export text lines."""

import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.*)$"
)
SYSTEM_EVENT_PATTERN = re.compile(
    r"^\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.+)$"
)
DELETED_MESSAGE_PATTERN = re.compile(
    r"^(?:This message was deleted\.?|Dit bericht is verwijderd\.?|"
    r"Je hebt dit bericht verwijderd\.?)$",
    flags=re.IGNORECASE,
)


def _strip_invisible(text: str) -> str:
    """Remove zero-width and control characters from text."""
    return text.replace("\u200e", "").replace("\u202f", "").strip()


def normalize_sender(sender: str) -> str:
    """Normalize sender names by trimming and standardizing whitespace."""
    return sender.strip()


def extract_emojis(text: str) -> list[str]:
    """Extract emojis from text using the emoji library."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.findall(text)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw WhatsApp export lines into structured messages.

    Produces:
      - datetime
      - sender
      - original_message
      - message
      - emoji_list
      - contains_emoji
    """

    logger.info("Starting data cleaning process")

    if "raw" not in df.columns:
        raise ValueError("Expected column 'raw' in raw dataframe")

    messages = []
    current = None

    for line in df["raw"]:

        line = "" if line is None else str(line).rstrip("\n")
        stripped = _strip_invisible(line)

        match = MESSAGE_PATTERN.match(stripped)

        if match:

            if current is not None:
                messages.append(current)

            date, time, sender, msg = match.groups()

            current = {
                "datetime": f"{date} {time}",
                "sender": normalize_sender(sender),
                "original_message": (msg or "").strip(),
            }

        elif SYSTEM_EVENT_PATTERN.match(stripped):
            if current is not None:
                messages.append(current)
                current = None
            continue

        elif current is not None and stripped:

            current["original_message"] += "\n" + stripped

    if current is not None:
        messages.append(current)

    df_clean = pd.DataFrame(messages)

    if df_clean.empty:
        raise ValueError(
            "No messages parsed. Check MESSAGE_PATTERN against WhatsApp export format."
        )

    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df_clean["sender"] = df_clean["sender"].astype(str).map(normalize_sender)

    df_clean["original_message"] = (
        df_clean["original_message"]
        .astype(str)
        .map(_strip_invisible)
        .str.strip()
    )

    df_clean["message"] = df_clean["original_message"]

    # Remove non-content rows that should not be analyzed as real messages.
    df_clean = df_clean.loc[
        ~df_clean["message"].fillna("").astype(str).str.match(DELETED_MESSAGE_PATTERN)
    ].copy()

    df_clean["emoji_list"] = df_clean["message"].apply(extract_emojis)

    df_clean["contains_emoji"] = df_clean["emoji_list"].apply(lambda x: len(x) > 0)

    df_clean = df_clean.drop_duplicates(
        subset=["datetime", "sender", "original_message"]
    ).reset_index(drop=True)

    logger.info(f"Data cleaning completed ({len(df_clean)} messages)")

    return df_clean
