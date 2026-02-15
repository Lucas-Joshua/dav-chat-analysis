import pandas as pd
from pathlib import Path

EMOJI_FILE = Path("data/raw/emoji.csv")


def load_emoji_df() -> pd.DataFrame:
    if not EMOJI_FILE.exists():
        raise FileNotFoundError("emoji.csv not found in data/raw")

    return pd.read_csv(EMOJI_FILE)


def build_char_to_name_map() -> dict[str, str]:
    df = load_emoji_df()
    return dict(zip(df["char"], df["name"]))


def build_name_to_char_map() -> dict[str, str]:
    df = load_emoji_df()
    return dict(zip(df["name"], df["char"]))