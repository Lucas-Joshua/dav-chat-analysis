import logging
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
from src import config

logger = logging.getLogger(__name__)

def get_real_start_date() -> pd.Timestamp | None:
    metadata_path = config.METADATA_FILE

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    real_start = meta.get("real_start_date")
    return pd.to_datetime(real_start) if real_start else None

def plot_messages_per_user(df: pd.DataFrame, top_n: int = 15) -> None:
    if df.empty:
        return

    messages_per_user = df["sender"].value_counts()
    top_users = messages_per_user.head(top_n).copy()

    others_count = messages_per_user.iloc[top_n:].sum()
    if others_count > 0:
        top_users.loc["Other"] = others_count

    plt.figure(figsize=config.FIGURE_SIZE)
    top_users.plot(kind="bar")

    plt.title(f"Messages per user (Top {top_n})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = config.IMG_DIR / "messages_per_user.png"
    plt.savefig(output_path, dpi=config.DPI)
    plt.close()

def plot_messages_per_day(df: pd.DataFrame) -> None:
    if df.empty:
        return

    df_copy = df.copy()

    real_start = get_real_start_date()
    if real_start is not None:
        df_copy = df_copy[df_copy["datetime"] >= real_start]

    df_copy["date"] = df_copy["datetime"].dt.date
    messages_per_day = df_copy.groupby("date").size()

    plt.figure(figsize=config.FIGURE_SIZE)
    messages_per_day.plot()

    plt.title("Messages per day")
    plt.tight_layout()

    output_path = config.IMG_DIR / "messages_per_day.png"
    plt.savefig(output_path, dpi=config.DPI)
    plt.close()

def plot_different_type_links(df: pd.DataFrame) -> None:
    if df.empty:
        return

    df_copy = df.copy()
    df_copy = df_copy[df_copy["link_source"].notna()]

    if df_copy.empty:
        return

    # Split meerdere bronnen
    df_copy["link_source"] = df_copy["link_source"].str.split(", ")
    df_copy = df_copy.explode("link_source")

    # Tel
    link_counts = df_copy["link_source"].value_counts()

    # Top N + Overig
    top_n = 8
    if len(link_counts) > top_n:
        top = link_counts.iloc[:top_n]
        rest = link_counts.iloc[top_n:].sum()
        link_counts = top.copy()
        link_counts.loc["Overig"] = rest

    # Sorteer voor nette barh
    link_counts = link_counts.sort_values()

    plt.figure(figsize=config.FIGURE_SIZE)
    bars = plt.barh(link_counts.index, link_counts.values)

    # Percentage labels
    total = link_counts.sum()
    for i, value in enumerate(link_counts.values):
        percent = (value / total) * 100
        plt.text(value, i, f" {percent:.1f}%", va="center")

    plt.title("URL type distribution")
    plt.xlabel("Aantal links")
    plt.ylabel("Type URL")
    plt.tight_layout()

    output_path = config.IMG_DIR / "different_type_urls.png"
    plt.savefig(output_path, dpi=config.DPI)
    plt.close()
def create_visuals(df: pd.DataFrame) -> None:
    plot_messages_per_user(df)
    plot_messages_per_day(df)
    plot_different_type_links(df)