import pandas as pd
import logging
import re
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

def add_swearing_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean column indicating presence of swear words."""

    if "message" not in df.columns:
        raise ValueError("Column 'message' not found in DataFrame")

    df = df.copy()

    df["contains_swear"] = (
        df["message"]
        .fillna("")
        .str.contains(SWEAR_PATTERN, regex=True, case=False)
    )

    return df

def extract_domains(text: str) -> str | None:
    """Extract normalized domain names from text."""

    if not isinstance(text, str):
        return None

    urls = URL_PATTERN.findall(text)

    if not urls:
        return None

    domains = []

    for url in urls:
        if not url.startswith("http"):
            url = "http://" + url

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")
            root = domain.split(".")[0]

            if root.startswith("youtu"):
                root = "youtube"

            domains.append(root.capitalize())

        except Exception:
            continue

    return ", ".join(sorted(set(domains))) if domains else None

def add_link_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add column with detected link sources."""

    if "message" not in df.columns:
        raise ValueError("Column 'message' not found in DataFrame")

    df = df.copy()
    df["link_source"] = df["message"].apply(extract_domains)

    return df

def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""

    logger.info("Starting feature engineering")

    df = add_link_feature(df)
    df = add_swearing_feature(df)

    logger.info("Feature engineering completed")

    return df