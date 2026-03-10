from __future__ import annotations

import pandas as pd
import emoji
import re


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


def extract_emojis(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]

def add_emoji_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    msg_col = "message" if "message" in df.columns else "original_message"
    if msg_col not in df.columns:
        raise KeyError("No message column found.")

    df["emoji_list"] = df[msg_col].apply(extract_emojis)
    df["emoji_count"] = df["emoji_list"].apply(len)
    df["has_emoji"] = df["emoji_count"] > 0

    return df

def add_emoji_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "emoji_list" not in df.columns:
        raise KeyError("emoji_list column not found.")

    df = df.explode("emoji_list").dropna(subset=["emoji_list"])

    df["emoji_type"] = df["emoji_list"].map(EMOJI_CATEGORY_MAP)
    df["emoji_group"] = df["emoji_type"].map(CATEGORY_REDUCTION_MAP)
    df = df.dropna(subset=["emoji_group"])

    return df

def _detect_user_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    if "user" in df.columns:
        return "user"
    if "sender" in df.columns:
        return "sender"
    raise KeyError("No user column found.")

def get_top_emojis_per_user(
    df: pd.DataFrame,
    top_n: int = 5,
    user_col: str | None = None
) -> pd.DataFrame:

    user_col = _detect_user_col(df, preferred=user_col)

    if "emoji_list" not in df.columns:
        raise KeyError("emoji_list column not found.")

    exploded = df.explode("emoji_list").dropna(subset=["emoji_list"])
    if exploded.empty:
        return pd.DataFrame(columns=[user_col, "emoji_list", "count"])

    counts = (
        exploded
        .groupby([user_col, "emoji_list"])
        .size()
        .reset_index(name="count")
    )

    top_per_user = (
        counts
        .sort_values([user_col, "count"], ascending=[True, False])
        .groupby(user_col)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_per_user

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["date_only"] = df["datetime"].dt.date

    return df

def add_message_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add message length feature based on message text.
    """

    df = df.copy()

    df["message_length"] = df["message"].str.len()

    return df

def add_has_emoji_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature indicating whether a message contains at least one emoji.
    """

    df = df.copy()

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]",
        flags=re.UNICODE,
    )

    df["has_emoji"] = df["message"].astype(str).apply(
        lambda x: bool(emoji_pattern.search(x))
    )

    return df