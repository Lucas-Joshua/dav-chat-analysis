import re
import pandas as pd
import logging
import emoji

from src.utils.emoji_utils import build_char_to_name_map

logger = logging.getLogger(__name__)

CHAR_TO_NAME = build_char_to_name_map()

MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)$"
)

_INVISIBLE = ["\u202f", "\u200e", "\u200f", "\ufeff"]


def normalize_sender(sender: str) -> str:
    sender = "" if sender is None else str(sender)

    for ch in _INVISIBLE:
        sender = sender.replace(ch, "")

    sender = sender.replace("\xa0", " ")
    sender = " ".join(sender.split())

    return sender.strip()


def normalize_emojis(text: str) -> tuple[str, list[str]]:
    if not isinstance(text, str):
        return text, []

    emoji_names = []

    for match in emoji.emoji_list(text):
        char = match["emoji"]
        name = CHAR_TO_NAME.get(char)

        if name:
            text = text.replace(char, f" {name} ")
            emoji_names.append(name)

    return text.strip(), emoji_names


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning process")

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
                "original_message": msg.strip(),
            }

        elif current and stripped:
            current["original_message"] += "\n" + stripped

    if current:
        messages.append(current)

    df_clean = pd.DataFrame(messages)

    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    df_clean["sender"] = df_clean["sender"].astype(str)

    normalized_messages = []
    emoji_lists = []
    contains_flags = []

    for msg in df_clean["original_message"]:
        norm_text, found = normalize_emojis(msg)
        normalized_messages.append(norm_text)
        emoji_lists.append(found)
        contains_flags.append(len(found) > 0)

    df_clean["message"] = normalized_messages
    df_clean["emoji_list"] = emoji_lists
    df_clean["contains_emoji"] = contains_flags

    logger.info(f"Data cleaning completed ({len(df_clean)} messages)")

    return df_clean