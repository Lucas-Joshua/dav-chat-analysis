from __future__ import annotations

import logging
from pathlib import Path

from src import config
from src.logging_config import setup_logging

from src.modules.data_loader import load_raw_chat
from src.modules.data_cleaning import clean_data
from src.modules.anonymizer import apply_anonymization
from src.modules.feature_engineering import (
    add_emoji_features,
    add_emoji_category,
    get_top_emojis_per_user,
)
from src.modules.visualization import (
    plot_top_emojis_per_user_png,
    plot_emoji_heatmap_png,
    plot_emoji_type_per_user,
)


logger = logging.getLogger(__name__)


def run_pipeline(raw_path: str | Path | None = None) -> None:
    setup_logging()
    logger.info("Pipeline started")

    if raw_path is None:
        raw_path = config.RAW_DATA_FILE

    raw_path = Path(raw_path)

    if not raw_path.exists():
        logger.error(f"Raw data file not found: {raw_path}")
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    logger.info(f"Using raw data file: {raw_path}")

    logger.info("Loading raw chat data")
    df_raw = load_raw_chat(raw_path)

    logger.info(f"Loaded {len(df_raw)} rows")

    logger.info("Cleaning data")
    df = clean_data(df_raw)

    logger.info(f"Rows after cleaning: {len(df)}")

    logger.info("Applying anonymization")
    df = apply_anonymization(df)


    logger.info("Extracting emoji features")
    df = add_emoji_features(df)

    logger.info("Adding emoji categories")
    df = add_emoji_category(df)

    logger.info("Saving processed data")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = config.PROCESSED_DIR / "clean_chat_anonymized.parquet"
    csv_path = config.PROCESSED_DIR / "clean_chat_anonymized.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved parquet: {parquet_path}")
    logger.info(f"Saved csv: {csv_path}")

    logger.info("Generating visualizations")

    # Top emojis per user
    top_user = get_top_emojis_per_user(df, top_n=5)
    plot_top_emojis_per_user_png(
        top_user,
        out_path=config.IMG_DIR / "top_emojis_per_user.png",
    )
    logger.info("Saved: top_emojis_per_user.png")

    # Emoji heatmap
    plot_emoji_heatmap_png(
        df,
        out_path=config.IMG_DIR / "emoji_heatmap.png",
        top_n_emojis=10,
    )
    logger.info("Saved: emoji_heatmap.png")

    # Emoji group distribution
    plot_emoji_type_per_user(
        df,
        out_path=config.IMG_DIR / "emoji_type_per_user.png",
        top_users=10,
    )
    logger.info("Saved: emoji_type_per_user.png")

    logger.info("Pipeline finished successfully")