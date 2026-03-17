"""Pipeline orchestration from raw input to processed outputs and charts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd
from src import config
from src.modules.feature_engineering import (
    add_emoji_category,
    add_emoji_features,
    add_has_emoji_feature,
    add_incident_bow_features,
    add_message_length,
    add_time_features,
)
from src.modules.preprocessor import ChatPreprocessor
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


class PipelineRunner:
    """Run the end-to-end data pipeline with configurable visualization steps."""

    def __init__(
        self,
        visualization_selections: Mapping[str, bool] | None = None,
        preprocessor: ChatPreprocessor | None = None,
    ) -> None:
        """Initialize runner dependencies and runtime configuration."""
        self.logger = logging.getLogger(__name__)
        self.visualization_selections = dict(
            visualization_selections or VISUALIZATION_SELECTIONS
        )
        self.preprocessor = preprocessor or ChatPreprocessor()

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering steps required by downstream plots."""
        self.logger.info("Step 4/6: add features")
        df = add_emoji_features(df)
        df = add_emoji_category(df)
        df = add_time_features(df)
        df = add_message_length(df)
        df = add_has_emoji_feature(df)
        df = add_incident_bow_features(df)
        self.logger.info("Dataset shape after processing: %s", df.shape)
        return df

    def _save_processed_files(self, df: pd.DataFrame) -> None:
        """Save processed dataframe to parquet and CSV outputs."""
        self.logger.info("Step 5/6: save processed files")
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_DIR / "clean_chat_processed.parquet", index=False)
        df.to_csv(config.PROCESSED_DIR / "clean_chat_processed.csv", index=False)

    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """Render enabled visualizations to the output image directory."""
        self.logger.info("Step 6/6: generate visualizations")
        config.IMG_DIR.mkdir(parents=True, exist_ok=True)
        run_selected(df, self.visualization_selections, out_dir=config.IMG_DIR)

    def run(self, raw_path: str | Path) -> None:
        """Execute the full pipeline from raw input to visualizations."""
        enabled = [name for name, on in self.visualization_selections.items() if on]
        self.logger.info(
            "Pipeline config loaded. %d visualizations enabled.", len(enabled)
        )
        df = self.preprocessor.run(raw_path)
        df = self._add_features(df)
        self._save_processed_files(df)
        self._generate_visualizations(df)


def run_pipeline(raw_path: str | Path) -> None:
    """Run the default end-to-end data pipeline."""
    PipelineRunner().run(raw_path)
