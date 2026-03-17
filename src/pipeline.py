from __future__ import annotations

import logging
from pathlib import Path

from src import config
from src.modules import anonymizer, data_cleaning, data_loader
from src.modules.feature_engineering import (
    add_emoji_category,
    add_emoji_features,
    add_has_emoji_feature,
    add_incident_bow_features,
    add_message_length,
    add_time_features,
)
from src.visualizations import run_selected

VISUALIZATION_SELECTIONS = {
    "overall_emoji_distribution": False,
    "emoji_heatmap": False,
    "emoji_type_per_user": False,
    "emoji_usage_by_hour": False,
    "negative_reaction_concentration": False,
    "negative_reaction_diagnostic": False,
    "negative_reaction_scatter": False,
    "chat_activity_by_hour": False,
    "chat_activity_distribution": False,
    "response_time_suite": False,
    "incident_discussion_timeline": True,
    "incident_activity_correlation": True,
}


def run_pipeline(raw_path: str | Path) -> None:
    """Run full data pipeline without ML, using feature engineering + BOW."""
    logger = logging.getLogger(__name__)
    enabled = [name for name, on in VISUALIZATION_SELECTIONS.items() if on]
    logger.info("Pipeline config loaded. %d visualizations enabled.", len(enabled))

    logger.info("Step 1/6: load")
    df = data_loader.load_raw_chat(Path(raw_path))

    logger.info("Step 2/6: clean")
    df = data_cleaning.clean_data(df)

    logger.info("Step 3/6: anonymize")
    df = anonymizer.apply_anonymization(df)

    logger.info("Step 4/6: add features")
    df = add_emoji_features(df)
    df = add_emoji_category(df)
    df = add_time_features(df)
    df = add_message_length(df)
    df = add_has_emoji_feature(df)
    df = add_incident_bow_features(df)
    logger.info("Dataset shape after processing: %s", df.shape)

    logger.info("Step 5/6: save processed files")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_DIR / "clean_chat_processed.parquet", index=False)
    df.to_csv(config.PROCESSED_DIR / "clean_chat_processed.csv", index=False)

    logger.info("Step 6/6: generate visualizations")
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    run_selected(df, VISUALIZATION_SELECTIONS, out_dir=config.IMG_DIR)
