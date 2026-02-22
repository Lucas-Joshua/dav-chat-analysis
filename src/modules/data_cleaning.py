import re
import logging
import pandas as pd
import emoji

logger = logging.getLogger(__name__)

# Match WhatsApp lines like:
# [06-10-2024, 09:04:09] ~ Sander: message
# [30-10-2024, 13:42:26] Sabien Skydive Hilversum: message
MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}),\s*(\d{2}:\d{2}:\d{2})\]\s*(.*?):\s*(.*)$"
)

# Common invisible chars in WhatsApp exports
_INVISIBLE = ["\u202f", "\u200e", "\u200f", "\ufeff", "\u200b", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "‎"]


def _strip_invisible(s: str) -> str:
    s = "" if s is None else str(s)
    for ch in _INVISIBLE:
        s = s.replace(ch, "")
    # normalize weird spaces
    s = s.replace("\xa0", " ")
    s = " ".join(s.split())
    return s.strip()


def normalize_sender(sender: str) -> str:
    sender = _strip_invisible(sender)
    # WhatsApp sometimes prefixes with "~"
    if sender.startswith("~"):
        sender = sender.lstrip("~").strip()
    return sender


def extract_emojis(text: str) -> list[str]:
    """
    Robust emoji extraction (handles ZWJ sequences like 🤷‍♂️).
    Returns list of emoji characters, not names.
    """
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw WhatsApp export lines into structured messages.
    Keeps emojis as real emojis (no name conversion).
    Produces:
      - datetime (datetime64)
      - sender (str)
      - original_message (str)
      - message (str)  # cleaned text (still contains emojis)
      - emoji_list (list[str])
      - contains_emoji (bool)
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
            # flush previous
            if current:
                messages.append(current)

            date, time, sender, msg = match.groups()

            current = {
                "datetime": f"{date} {time}",
                "sender": normalize_sender(sender),
                "original_message": (msg or "").strip(),
            }

        elif current and stripped:
            # continuation of multi-line message
            current["original_message"] += "\n" + stripped

        else:
            # ignore stray lines before first message or empty lines
            continue

    if current:
        messages.append(current)

    df_clean = pd.DataFrame(messages)

    # Parse datetime
    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    # Basic text cleaning
    df_clean["sender"] = df_clean["sender"].astype(str).map(normalize_sender)
    df_clean["original_message"] = df_clean["original_message"].astype(str).map(_strip_invisible).str.strip()

    # Keep message as cleaned original (do NOT replace emojis)
    df_clean["message"] = df_clean["original_message"]

    # Emoji features
    df_clean["emoji_list"] = df_clean["message"].apply(extract_emojis)
    df_clean["contains_emoji"] = df_clean["emoji_list"].apply(lambda x: len(x) > 0)

    logger.info(f"Data cleaning completed ({len(df_clean)} messages)")
    return df_clean