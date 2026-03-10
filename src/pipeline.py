from __future__ import annotations

import logging
from pathlib import Path

from src import config
from src.logging_config import setup_logging
from src.modules import data_loader, data_cleaning
from src.modules.feature_engineering import (
    add_emoji_features,
    add_emoji_category,
    add_time_features,
    add_message_length,
    add_has_emoji_feature
)
from src.modules.visualization import (
    plot_overall_emoji_distribution,
    plot_negative_reaction_concentration,
    plot_negative_reaction_scatter,
    plot_chat_activity_by_hour,
    plot_chat_activity_distribution,
    plot_emoji_usage_by_hour
)

# ==================================================
# VISUALIZATION TOGGLES
# ==================================================
GENERATE_OVERALL_DISTRIBUTION: bool = False
GENERATE_NEGATIVE_CONCENTRATION: bool = False
GENERATE_NEGATIVE_SCATTER: bool = False
GENERATE_HOURLY_ACTIVITY: bool = False
GENERATE_MESSAGE_LENGTH_DISTRIBUTION: bool = False
GENERATE_EMOJI_USAGE_BY_HOUR: bool = True

# ==================================================


def run_pipeline(raw_path: str | Path) -> None:
    """
    Main pipeline execution.

    Steps:
    1. Load raw chat data
    2. Clean dataset
    3. Add engineered features
    4. Save processed dataset
    5. Generate selected visualizations
    """

    # -------------------------
    # Setup logging
    # -------------------------
    setup_logging(log_to_file=True)
    logger = logging.getLogger(__name__)

    logger.info("========== PIPELINE STARTED ==========")

    raw_path = Path(raw_path)

    if not raw_path.exists():
        logger.error(f"Raw file not found: {raw_path}")
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # Ensure output directories exist
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load
    # -------------------------
    logger.info("Loading raw chat data...")
    df_raw = data_loader.load_raw_chat(raw_path)

    # -------------------------
    # Clean
    # -------------------------
    logger.info("Cleaning data...")
    df = data_cleaning.clean_data(df_raw)

    # -------------------------
    # Feature engineering
    # -------------------------
    logger.info("Adding emoji features...")
    df = add_emoji_features(df)

    logger.info("Adding emoji categories...")
    df = add_emoji_category(df)

    logger.info("Adding time features...")
    df = add_time_features(df)

    logger.info("Adding message length feature...")
    df = add_message_length(df)

    logger.info("Adding has emoji feature...")
    df = add_has_emoji_feature(df)

    logger.info(f"Dataset shape after processing: {df.shape}")

    # -------------------------
    # Save processed dataset
    # -------------------------
    processed_path = config.PROCESSED_DIR / "clean_chat_processed.parquet"
    logger.info(f"Saving processed dataset to: {processed_path}")
    df.to_parquet(processed_path, index=False)

    # -------------------------
    # Visualizations
    # -------------------------
    logger.info("Generating selected visualizations...")

    if GENERATE_OVERALL_DISTRIBUTION:
        logger.info("→ Overall emoji distribution")
        plot_overall_emoji_distribution(
            df,
            out_path=config.IMG_DIR / "overall_emoji_distribution.png",
        )

    if GENERATE_NEGATIVE_CONCENTRATION:
        logger.info("→ Negative reaction concentration")
        plot_negative_reaction_concentration(
            df,
            out_path=config.IMG_DIR / "negative_reaction_concentration.png",
        )

    if GENERATE_NEGATIVE_SCATTER:
        logger.info("→ Negative reaction scatter diagnostic")
        plot_negative_reaction_scatter(
            df,
            out_path=config.IMG_DIR / "negative_reaction_scatter.png",
        )

    if GENERATE_HOURLY_ACTIVITY:
        logger.info("→ Hourly activity plot")
        plot_chat_activity_by_hour(
            df,
            out_path=config.IMG_DIR / "chat_activity_by_hour.png",
        )

    if GENERATE_MESSAGE_LENGTH_DISTRIBUTION:
        logger.info("→ Message length distribution plot")

        plot_chat_activity_distribution(
            df,
            config.IMG_DIR / "plot_chat_activity_distribution.png",
        )

    if GENERATE_EMOJI_USAGE_BY_HOUR:
        logger.info("→ Emoji usage by hour plot")

        plot_emoji_usage_by_hour(
            df,
            config.IMG_DIR / "plot_emoji_usage_by_hour.png",
        )


    logger.info("========== PIPELINE FINISHED SUCCESSFULLY ==========")