"""Cleaning and parsing logic for WhatsApp export text lines."""

import pandas as pd
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

DATE_PATTERN = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
TIME_PATTERN = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?"

MESSAGE_PATTERNS = (
    re.compile(
        rf"^\[(?P<date>{DATE_PATTERN}), (?P<time>{TIME_PATTERN})\] (?P<sender>[^:]+): (?P<message>.*)$"
    ),
    re.compile(
        rf"^(?P<date>{DATE_PATTERN}), (?P<time>{TIME_PATTERN}) - (?P<sender>[^:]+): (?P<message>.*)$"
    ),
    re.compile(
        rf"^(?P<date>{DATE_PATTERN}) (?P<time>{TIME_PATTERN}) - (?P<sender>[^:]+): (?P<message>.*)$"
    ),
    re.compile(
        rf"^(?P<date>{DATE_PATTERN}) - (?P<sender>[^:]+): (?P<message>.*)$"
    ),
)
SYSTEM_EVENT_PATTERNS = (
    re.compile(rf"^\[(?P<date>{DATE_PATTERN}), (?P<time>{TIME_PATTERN})\] (?P<message>.+)$"),
    re.compile(rf"^(?P<date>{DATE_PATTERN}), (?P<time>{TIME_PATTERN}) - (?P<message>.+)$"),
    re.compile(rf"^(?P<date>{DATE_PATTERN}) (?P<time>{TIME_PATTERN}) - (?P<message>.+)$"),
    re.compile(rf"^(?P<date>{DATE_PATTERN}) - (?P<message>.+)$"),
)
DELETED_MESSAGE_PATTERN = re.compile(
    r"^(?:This message was deleted(?:\s+by\s+admin\s+\S+)?\.?|"
    r"Dit bericht is verwijderd\.?|"
    r"Je hebt dit bericht verwijderd\.?)$",
    flags=re.IGNORECASE,
)

# WhatsApp system notifications that slip through as regular messages
# (group membership events, encryption notices, etc.)
SYSTEM_CONTENT_PATTERN = re.compile(
    r"^(?:"
    r".+\s+was added"
    r"|.+\s+were added"
    r"|.+\s+added\s+.+"
    r"|.+\s+left"
    r"|.+\s+joined using this group(?:'s|s) invite link"
    r"|.+\s+joined via invite link"
    r"|.+\s+changed the group (?:name|description|icon|settings)"
    r"|.+\s+changed this group's icon"
    r"|Messages and calls are end-to-end encrypted.*"
    r"|Berichten en gesprekken zijn beveiligd.*"
    r"|You were added"
    r"|You changed the subject.*"
    r"|You changed this group.*"
    r"|\+\d[\d\s]+ was added"
    r"|image omitted"
    r"|video omitted"
    r"|audio omitted"
    r"|sticker omitted"
    r"|GIF omitted"
    r"|document omitted"
    r"|Contact card omitted"
    r"|<Media omitted>"
    r")$",
    flags=re.IGNORECASE,
)


def _strip_invisible(text: str) -> str:
    """Remove zero-width and control characters from text.

    :param text: Input text.
    :type text: str
    :return: Cleaned text without configured invisible characters.
    :rtype: str
    """
    return text.replace("\u200e", "").replace("\u202f", "").strip()


def normalize_sender(sender: str) -> str:
    """Normalize sender names by trimming surrounding whitespace.

    :param sender: Raw sender value.
    :type sender: str
    :return: Normalized sender name.
    :rtype: str
    """
    return sender.strip()


def extract_emojis(text: str) -> list[str]:
    """Extract emojis from text using a Unicode range pattern.

    :param text: Input message text.
    :type text: str
    :return: Emoji matches found in the text.
    :rtype: list[str]
    """
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


def _normalize_text_columns(
    df: pd.DataFrame,
    columns: list[str],
    normalizer: Callable[[str], str],
) -> pd.DataFrame:
    """Normalize multiple text columns using the same transformation function.

    :param df: Input dataframe.
    :type df: pd.DataFrame
    :param columns: Column names to normalize.
    :type columns: list[str]
    :param normalizer: Function applied to each string value.
    :type normalizer: Callable[[str], str]
    :return: Dataframe copy with normalized text columns.
    :rtype: pd.DataFrame
    """
    working = df.copy()
    for column in columns:
        if column not in working.columns:
            continue
        working[column] = working[column].astype(str).map(normalizer).str.strip()
    return working


def _match_line(patterns: tuple[re.Pattern[str], ...], text: str) -> re.Match[str] | None:
    """Try multiple regex patterns and return the first match."""
    for pattern in patterns:
        match = pattern.match(text)
        if match:
            return match
    return None


def _parse_chat_datetime(series: pd.Series) -> pd.Series:
    """Parse WhatsApp timestamps from multiple export formats.

    Supports common exports from Android, iPhone, Mac, and Windows by allowing:
    - bracketed or unbracketed prefixes
    - slash or hyphen date separators
    - 24-hour or 12-hour times
    - optional seconds
    """
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            errors="coerce",
            dayfirst=False,
        )
    return parsed


def _combine_date_and_time(date_text: str, time_text: str | None) -> str:
    """Build one datetime string, defaulting missing times to midnight."""
    if time_text is None or not str(time_text).strip():
        return f"{date_text} 00:00:00"
    return f"{date_text} {time_text}"


def _build_boolean_feature(series: pd.Series) -> pd.Series:
    """Build a boolean feature from list-like values based on non-empty length.

    :param series: Input series containing list-like values.
    :type series: pd.Series
    :return: Boolean series where ``True`` means a non-empty list.
    :rtype: pd.Series
    """
    return series.apply(lambda value: len(value) > 0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw WhatsApp export lines into structured messages.

    :param df: Raw dataframe containing a ``raw`` text column.
    :type df: pd.DataFrame
    :return: Cleaned dataframe with parsed metadata and emoji features.
    :rtype: pd.DataFrame
    """

    logger.info("Starting data cleaning process")

    if "raw" not in df.columns:
        raise ValueError("Expected column 'raw' in raw dataframe")

    messages = []
    current = None

    for line in df["raw"]:

        line = "" if line is None else str(line).rstrip("\n")
        stripped = _strip_invisible(line)

        match = _match_line(MESSAGE_PATTERNS, stripped)

        if match:

            if current is not None:
                messages.append(current)

            date = match.group("date")
            time = match.groupdict().get("time")
            sender = match.group("sender")
            msg = match.group("message")

            current = {
                "datetime": _combine_date_and_time(date, time),
                "sender": normalize_sender(sender),
                "original_message": (msg or "").strip(),
            }

        elif _match_line(SYSTEM_EVENT_PATTERNS, stripped):
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

    df_clean["datetime"] = _parse_chat_datetime(df_clean["datetime"])
    df_clean = _normalize_text_columns(
        df_clean,
        columns=["sender"],
        normalizer=normalize_sender,
    )
    df_clean = _normalize_text_columns(
        df_clean,
        columns=["original_message"],
        normalizer=_strip_invisible,
    )

    df_clean["message"] = df_clean["original_message"]

    msg_text = df_clean["message"].fillna("").astype(str)
    df_clean = df_clean.loc[
        ~msg_text.str.match(DELETED_MESSAGE_PATTERN)
        & ~msg_text.str.match(SYSTEM_CONTENT_PATTERN)
    ].copy()

    removed_system = msg_text.str.match(SYSTEM_CONTENT_PATTERN).sum()
    if removed_system:
        logger.info("Removed %d WhatsApp system notification messages", removed_system)

    df_clean["emoji_list"] = df_clean["message"].apply(extract_emojis)
    df_clean["contains_emoji"] = _build_boolean_feature(df_clean["emoji_list"])

    df_clean = df_clean.drop_duplicates(
        subset=["datetime", "sender", "original_message"]
    ).reset_index(drop=True)

    logger.info(f"Data cleaning completed ({len(df_clean)} messages)")

    return df_clean
