import logging
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
import config

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Helper: real start date uit metadata
# --------------------------------------------------
def get_real_start_date() -> pd.Timestamp | None:
    metadata_path = config.OUTPUT_DIR / "metadata.json"

    if not metadata_path.exists():
        logger.warning("metadata.json niet gevonden, volledige dataset gebruikt")
        return None

    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        real_start = meta.get("real_start_date")
        if real_start:
            return pd.to_datetime(real_start)

    except Exception as e:
        logger.warning(f"Kon metadata niet lezen: {e}")

    return None


# --------------------------------------------------
# Helper: emojis strippen voor labels
# --------------------------------------------------
def strip_emojis(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

# --------------------------------------------------
# Visual 1: Berichten per gebruiker
# --------------------------------------------------
def plot_messages_per_user(df: pd.DataFrame, top_n: int = 15) -> None:
    logger.info("Visual: berichten per gebruiker")

    messages_per_user = df["sender"].value_counts()

    # Alleen top N
    top_users = messages_per_user.head(top_n)

    # Rest samenvoegen
    others_count = messages_per_user.iloc[top_n:].sum()
    if others_count > 0:
        top_users["Overige"] = others_count

    plt.figure(figsize=(12, 6))
    top_users.plot(kind="bar")

    plt.title(f"Aantal berichten per gebruiker (Top {top_n})")
    plt.xlabel("Gebruiker")
    plt.ylabel("Aantal berichten")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / "messages_per_user.png"
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Visual opgeslagen: {output_path}")

# --------------------------------------------------
# Visual 2: Berichten per dag
# --------------------------------------------------
def plot_messages_per_day(df: pd.DataFrame) -> None:
    logger.info("Visual: berichten per dag")

    df_copy = df.copy()

    # Filter vanaf echte startdatum
    real_start = get_real_start_date()
    if real_start is not None:
        df_copy = df_copy[df_copy["datetime"] >= real_start]

    df_copy["date"] = df_copy["datetime"].dt.date
    messages_per_day = df_copy.groupby("date").size()

    plt.figure(figsize=(12, 6))
    messages_per_day.plot()
    plt.title("Aantal berichten per dag")
    plt.xlabel("Datum")
    plt.ylabel("Aantal berichten")
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / "messages_per_day.png"
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Visual opgeslagen: {output_path}")


# --------------------------------------------------
# Orchestrator
# --------------------------------------------------
def create_visuals(df: pd.DataFrame) -> None:
    logger.info("Start maken van visualisaties")

    plot_messages_per_user(df)
    plot_messages_per_day(df)

    logger.info("Visualisaties afgerond")