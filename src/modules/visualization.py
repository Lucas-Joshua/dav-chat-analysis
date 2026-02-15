import logging
from pathlib import Path
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src import config

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load emoji metadata (PNG source)
# --------------------------------------------------

EMOJI_FILE = Path("data/raw/emoji.csv")

if EMOJI_FILE.exists():
    EMOJI_DF = pd.read_csv(EMOJI_FILE)
else:
    EMOJI_DF = pd.DataFrame()


# --------------------------------------------------
# Helper: Add emoji image to plot (OS independent)
# --------------------------------------------------

def add_emoji_image(ax, x, y, url, zoom=0.06):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))

        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)

        ax.add_artist(ab)

    except Exception:
        pass


# --------------------------------------------------
# Messages per user
# --------------------------------------------------

def plot_messages_per_user(df: pd.DataFrame):
    counts = df["sender"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Messages per User")
    ax.set_ylabel("Messages")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output = config.IMG_DIR / "messages_per_user.png"
    plt.savefig(output, dpi=config.DPI)
    plt.close()

    logger.info(f"Saved: {output}")


# --------------------------------------------------
# Messages per day
# --------------------------------------------------

def plot_messages_per_day(df: pd.DataFrame):
    daily = df.groupby(df["datetime"].dt.date).size()

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    daily.plot(ax=ax)

    ax.set_title("Messages per Day")
    ax.set_ylabel("Messages")

    plt.tight_layout()

    output = config.IMG_DIR / "messages_per_day.png"
    plt.savefig(output, dpi=config.DPI)
    plt.close()

    logger.info(f"Saved: {output}")


# --------------------------------------------------
# URL types
# --------------------------------------------------

def plot_url_types(df: pd.DataFrame):
    if "link_source" not in df.columns:
        return

    counts = df["link_source"].value_counts()

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Different URL Types")
    ax.set_ylabel("Count")

    plt.tight_layout()

    output = config.IMG_DIR / "different_type_urls.png"
    plt.savefig(output, dpi=config.DPI)
    plt.close()

    logger.info(f"Saved: {output}")


# --------------------------------------------------
# Emoji usage per user
# --------------------------------------------------

def plot_emoji_usage_per_user(df: pd.DataFrame):
    if "contains_emoji" not in df.columns:
        return

    usage = df[df["contains_emoji"]].groupby("sender").size().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    usage.plot(kind="bar", ax=ax)

    ax.set_title("Emoji Usage per User")
    ax.set_ylabel("Emoji Messages")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = config.IMG_DIR / "emoji_usage_per_user.png"
    plt.savefig(output, dpi=config.DPI)
    plt.close()

    logger.info(f"Saved: {output}")


# --------------------------------------------------
# Top emojis (OS-independent PNG method)
# --------------------------------------------------
def plot_top_emojis(df: pd.DataFrame, top_n: int = 10):
    if "emoji_list" not in df.columns:
        return

    all_emojis = []

    for emojis in df["emoji_list"]:
        if isinstance(emojis, list):
            all_emojis.extend(emojis)

    if not all_emojis:
        return

    emoji_counts = pd.Series(all_emojis).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    counts = emoji_counts.values
    names = emoji_counts.index.tolist()

    y_positions = list(range(len(counts)))

    ax.barh(y_positions, counts, color="#2E6F9E")

    # 👇 Zorg voor ruimte links
    max_count = max(counts)
    ax.set_xlim(-max_count * 0.25, max_count * 1.05)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(y_positions))

    # 👇 Emoji images toevoegen
    for i, (name, count) in enumerate(zip(names, counts)):
        row = EMOJI_DF[EMOJI_DF["name"].str.lower() == name.lower()]

        if not row.empty:
            url = row.iloc[0]["url"]

            # Plaats afbeelding duidelijk links van bar
            add_emoji_image(ax, -max_count * 0.15, i, url)

    ax.set_xlabel("Frequency")
    ax.set_title("Top Emojis")
    ax.invert_yaxis()

    plt.tight_layout()

    output = config.IMG_DIR / "top_emojis.png"
    plt.savefig(output, dpi=config.DPI)
    plt.close()

    print(names)
    print(EMOJI_DF["name"].head())

    logger.info(f"Saved: {output}")
# --------------------------------------------------
# Main entry
# --------------------------------------------------

def create_visualizations(df: pd.DataFrame):
    logger.info("Start creating visualizations")

    plot_messages_per_user(df)
    plot_messages_per_day(df)
    plot_url_types(df)
    plot_emoji_usage_per_user(df)
    plot_top_emojis(df)

    logger.info("Visualizations completed")