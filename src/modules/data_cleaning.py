import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)$"
)

_INVISIBLE = [
    "\u202f",
    "\u200e",
    "\u200f",
    "\ufeff",
]


def normalize_sender(sender: str) -> str:
    """Normalize sender names by removing invisible characters and excess whitespace."""
    sender = "" if sender is None else str(sender)

    sender = sender.replace("~", "")

    for ch in _INVISIBLE:
        sender = sender.replace(ch, "")

    sender = sender.replace("\xa0", " ")
    sender = " ".join(sender.split())

    return sender.strip()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw WhatsApp export into structured DataFrame
    with columns: datetime, sender, message.
    """

    logger.info("Starting data cleaning process")

    if "raw" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'raw' column")

    messages = []
    current = None

    for line in df["raw"]:
        line = "" if line is None else str(line).rstrip("\n")
        stripped = line.strip()

        match = MESSAGE_PATTERN.match(stripped)

        if match:
            if current:
                messages.append(current)

            date, time, sender, msg = match.groups()

            current = {
                "datetime": f"{date} {time}",
                "sender": normalize_sender(sender),
                "message": msg.strip(),
            }

        elif current and stripped:
            current["message"] += "\n" + stripped

    if current:
        messages.append(current)

    if not messages:
        logger.warning("No valid messages detected in chat export")
        return pd.DataFrame(columns=["datetime", "sender", "message"])

    df_clean = pd.DataFrame(messages)

    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    if df_clean["datetime"].isna().any():
        logger.warning("Some datetime values could not be parsed")

    df_clean["sender"] = df_clean["sender"].astype(str).map(normalize_sender)

    logger.info(f"Data cleaning completed ({len(df_clean)} messages)")

    return df_clean