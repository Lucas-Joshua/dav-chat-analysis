"""Feature engineering steps for emoji, time, length, and incident signals."""

from __future__ import annotations

import re
import pandas as pd
import emoji


EMOJI_CATEGORY_MAP = {
    "😄": "positive", "😁": "positive", "🙂": "positive",
    "😊": "positive", "😃": "positive", "😍": "positive",
    "🥰": "positive", "😇": "positive", "😌": "positive",
    "🌟": "positive",

    "😂": "humor", "🤣": "humor", "😹": "humor",
    "😆": "humor", "😜": "humor", "😝": "humor",
    "🤪": "humor",

    "😎": "cool", "🙃": "cool", "😏": "cool",
    "😬": "cool", "🫠": "cool",

    "🙏": "support", "👍": "support", "👏": "support",
    "🙌": "support", "💪": "support", "🤝": "support",

    "🤔": "reflective", "😅": "reflective",
    "😐": "reflective", "😶": "reflective",
    "🧐": "reflective",

    "🎉": "celebration", "🥳": "celebration",
    "🎊": "celebration",

    "❤️": "affection", "💖": "affection",
    "💙": "affection", "💛": "affection",
    "💕": "affection", "😘": "affection",

    "😒": "negative", "😤": "negative",
    "😩": "negative", "😢": "negative",
    "😭": "negative", "😔": "negative",
}


CATEGORY_REDUCTION_MAP = {
    "positive": "positive",
    "celebration": "positive",
    "affection": "positive",
    "support": "social",
    "humor": "humor",
    "cool": "humor",
    "negative": "negative_reflective",
    "reflective": "negative_reflective",
}


INCIDENT_BOW_TERMS = [
    "reserve",
    "reserve ride",
    "cutaway",
    "malfunction",
    "line twist",
    "hard opening",
    "low turn",
    "two out",
    "canopy collision",
    "bag lock",
    "horseshoe",
    "pilot chute in tow",
    "storing",
    "lijn twist",
    "harde opening",
    "botsing",
    "noodparachute",
    "afwerp",
    "afwerp hendel",
    "incident",
    "ongeval",
    "gewond",
    "fatal",
]


def _build_incident_pattern() -> str:
    """Build a regex pattern that matches configured incident bag-of-words terms.

    :return: Regex pattern string for incident keyword matching.
    :rtype: str
    """
    escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in INCIDENT_BOW_TERMS]
    return r"\b(?:{})\b".format("|".join(escaped))


def extract_emojis(text: str) -> list[str]:
    """Extract a list of emojis from text.

    :param text: Input message text.
    :type text: str
    :return: Extracted emoji characters.
    :rtype: list[str]
    """
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


def add_emoji_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji list, count, and presence features to the dataset.

    :param df: Input dataframe with a message column.
    :type df: pd.DataFrame
    :return: Dataframe with ``emoji_list``, ``emoji_count``, and ``has_emoji``.
    :rtype: pd.DataFrame
    """
    df = df.copy()

    msg_col = "message" if "message" in df.columns else "original_message"

    if msg_col not in df.columns:
        raise KeyError("No message column found.")

    df["emoji_list"] = df[msg_col].apply(extract_emojis)
    df["emoji_count"] = df["emoji_list"].apply(len)
    df["has_emoji"] = df["emoji_count"] > 0

    return df


def _determine_emoji_category(emojis: list[str]) -> tuple[str | None, str | None]:
    """Map the first recognized emoji to a type and reduced group.

    :param emojis: Emoji list extracted from a message.
    :type emojis: list[str]
    :return: Tuple of ``(emoji_type, emoji_group)`` or ``(None, None)``.
    :rtype: tuple[str | None, str | None]
    """

    if not emojis:
        return None, None

    for e in emojis:
        emoji_type = EMOJI_CATEGORY_MAP.get(e)

        if emoji_type:
            emoji_group = CATEGORY_REDUCTION_MAP.get(emoji_type)
            return emoji_type, emoji_group

    return None, None


def add_emoji_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji type and reduced group columns based on emoji list.

    :param df: Input dataframe containing ``emoji_list``.
    :type df: pd.DataFrame
    :return: Dataframe with ``emoji_type`` and ``emoji_group`` columns.
    :rtype: pd.DataFrame
    """

    df = df.copy()

    if "emoji_list" not in df.columns:
        raise KeyError("emoji_list column not found.")

    categories = df["emoji_list"].apply(_determine_emoji_category)

    df["emoji_type"] = categories.apply(lambda x: x[0])
    df["emoji_group"] = categories.apply(lambda x: x[1])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day name, and date-only columns from datetime.

    :param df: Input dataframe containing ``datetime``.
    :type df: pd.DataFrame
    :return: Dataframe with time-derived columns.
    :rtype: pd.DataFrame
    """

    df = df.copy()

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")

    datetime_series = pd.to_datetime(df["datetime"], errors="coerce")
    df["datetime"] = datetime_series
    df["hour"] = datetime_series.map(
        lambda value: int(value.hour) if pd.notna(value) else None
    )
    df["day_of_week"] = datetime_series.map(
        lambda value: value.strftime("%A") if pd.notna(value) else None
    )
    df["date_only"] = datetime_series.map(
        lambda value: value.date() if pd.notna(value) else None
    )

    return df


def add_message_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add message length in characters.

    :param df: Input dataframe containing ``message``.
    :type df: pd.DataFrame
    :return: Dataframe with ``message_length``.
    :rtype: pd.DataFrame
    """

    df = df.copy()

    df["message_length"] = df["message"].astype(str).str.len()

    return df


def add_has_emoji_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag indicating emoji presence.

    :param df: Input dataframe containing ``emoji_count``.
    :type df: pd.DataFrame
    :return: Dataframe with refreshed ``has_emoji`` flag.
    :rtype: pd.DataFrame
    """
    df = df.copy()

    if "emoji_count" not in df.columns:
        raise KeyError("emoji_count column not found. Run add_emoji_features first.")

    df["has_emoji"] = df["emoji_count"] > 0

    return df


def add_incident_bow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bag-of-words incident features without a machine-learning model.

    :param df: Input dataframe containing ``message``.
    :type df: pd.DataFrame
    :return: Dataframe with incident hit, score, and flag columns.
    :rtype: pd.DataFrame
    """
    df = df.copy()
    if "message" not in df.columns:
        raise KeyError("message column not found.")

    pattern = _build_incident_pattern()
    text = df["message"].fillna("").astype(str)

    df["incident_bow_hits"] = text.str.count(pattern, flags=re.IGNORECASE).fillna(0).astype(int)
    df["incident_bow_score"] = (df["incident_bow_hits"] / (df["incident_bow_hits"] + 1)).astype(float)
    df["is_incident_message"] = (df["incident_bow_hits"] >= 1).astype(int)
    return df
