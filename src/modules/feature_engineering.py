import pandas as pd
import logging
import re
import emoji
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


SWEAR_WORDS = [
    "kut",
    "fuck",
    "shit",
    "tering",
    "klootzak",
    "godver",
]

SWEAR_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(map(re.escape, SWEAR_WORDS))
    + r")\b",
    flags=re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    flags=re.IGNORECASE,
)


# --------------------------------------------------
# Swearing
# --------------------------------------------------

def add_swearing_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "message" not in df.columns:
        raise ValueError("Column 'message' not found in DataFrame")

    df = df.copy()

    df["contains_swear"] = (
        df["message"]
        .fillna("")
        .str.contains(SWEAR_PATTERN)
    )

    return df


# --------------------------------------------------
# Links
# --------------------------------------------------

def extract_domains(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    urls = URL_PATTERN.findall(text)
    if not urls:
        return None

    domains = []

    for url in urls:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")
            root = domain.split(".")[0]

            if root.startswith("youtu"):
                root = "youtube"

            domains.append(root.capitalize())

        except ValueError:
            logger.debug(f"Invalid URL skipped: {url}")
            continue

    return ", ".join(sorted(set(domains))) if domains else None


def add_link_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "message" not in df.columns:
        raise ValueError("Column 'message' not found in DataFrame")

    df = df.copy()
    df["link_source"] = df["message"].apply(extract_domains)

    return df


# --------------------------------------------------
# Emojis
# --------------------------------------------------

def extract_emojis(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    return [char for char in text if char in emoji.EMOJI_DATA]


def add_emoji_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "message" not in df.columns:
        raise ValueError("Column 'message' not found in DataFrame")

    df = df.copy()

    df["emoji_list"] = df["message"].apply(extract_emojis)
    df["contains_emoji"] = df["emoji_list"].str.len() > 0

    return df


def get_top_emojis(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    if "emoji_list" not in df.columns:
        raise ValueError("Column 'emoji_list' not found in DataFrame")

    return (
        df["emoji_list"]
        .explode()
        .dropna()
        .value_counts()
        .head(top_n)
    )


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering")

    df = add_link_feature(df)
    df = add_swearing_feature(df)
    df = add_emoji_feature(df)

    logger.info("Feature engineering completed")

    return df