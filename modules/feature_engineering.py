import pandas as pd
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =========================================================
# SWEAR FEATURE
# =========================================================

SWEAR_WORDS = [
    "kut",
    "fuck",
    "shit",
    "tering",
    "klootzak",
    "godver",
]

SWEAR_PATTERN = re.compile(
    r"\b(" + "|".join(SWEAR_WORDS) + r")\b",
    flags=re.IGNORECASE,
)

def add_swearing_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def contains_swear(text):
        if not isinstance(text, str):
            return False
        return bool(SWEAR_PATTERN.search(text))

    df["contains_swear"] = df["message"].apply(contains_swear)

    return df

# =========================================================
# LINK FEATURE
# =========================================================

URL_PATTERN = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    flags=re.IGNORECASE,
)

def extract_domains(text):
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
            domain = parsed.netloc.lower()

            domain = domain.replace("www.", "")
            root = domain.split(".")[0]

            # Alleen kleine uitzonderingen
            if root.startswith("youtu"):
                root = "youtube"

            domains.append(root.capitalize())

        except Exception:
            continue

    if not domains:
        return None

    return ", ".join(sorted(set(domains)))
def add_link_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["link_source"] = df["message"].apply(extract_domains)
    return df

# =========================================================
# MAIN FEATURE PIPELINE
# =========================================================

def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline.
    """

    logger.info("Start feature engineering")

    df = add_link_feature(df)
    df = add_swearing_feature(df)

    logger.info("Feature engineering afgerond")

    return df