from __future__ import annotations

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