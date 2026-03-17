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
    escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in INCIDENT_BOW_TERMS]
    return r"\b(?:{})\b".format("|".join(escaped))


def extract_emojis(text: str) -> list[str]:
    """Extract a list of emojis from text."""
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


def add_emoji_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji list, count, and presence features to the dataset."""
    df = df.copy()

    msg_col = "message" if "message" in df.columns else "original_message"

    if msg_col not in df.columns:
        raise KeyError("No message column found.")

    df["emoji_list"] = df[msg_col].apply(extract_emojis)
    df["emoji_count"] = df["emoji_list"].apply(len)
    df["has_emoji"] = df["emoji_count"] > 0

    return df


def _determine_emoji_category(emojis: list[str]) -> tuple[str | None, str | None]:
    """Map the first recognized emoji to a type and reduced group."""

    if not emojis:
        return None, None

    for e in emojis:
        emoji_type = EMOJI_CATEGORY_MAP.get(e)

        if emoji_type:
            emoji_group = CATEGORY_REDUCTION_MAP.get(emoji_type)
            return emoji_type, emoji_group

    return None, None


def add_emoji_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji type and reduced group columns based on emoji list."""

    df = df.copy()

    if "emoji_list" not in df.columns:
        raise KeyError("emoji_list column not found.")

    categories = df["emoji_list"].apply(_determine_emoji_category)

    df["emoji_type"] = categories.apply(lambda x: x[0])
    df["emoji_group"] = categories.apply(lambda x: x[1])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day name, and date-only columns from datetime."""

    df = df.copy()

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["date_only"] = df["datetime"].dt.date

    return df


def add_message_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add message length in characters."""

    df = df.copy()

    df["message_length"] = df["message"].astype(str).str.len()

    return df


def add_has_emoji_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag indicating emoji presence."""
    df = df.copy()

    if "emoji_count" not in df.columns:
        raise KeyError("emoji_count column not found. Run add_emoji_features first.")

    df["has_emoji"] = df["emoji_count"] > 0

    return df


def add_incident_bow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bag-of-words incident features (no ML model)."""
    df = df.copy()
    if "message" not in df.columns:
        raise KeyError("message column not found.")

    pattern = _build_incident_pattern()
    text = df["message"].fillna("").astype(str)

    df["incident_bow_hits"] = text.str.count(pattern, flags=re.IGNORECASE).fillna(0).astype(int)
    # Simple normalized score in [0, 1] from hit count.
    df["incident_bow_score"] = (df["incident_bow_hits"] / (df["incident_bow_hits"] + 1)).astype(float)
    df["is_incident_message"] = (df["incident_bow_hits"] >= 1).astype(int)
    return df
