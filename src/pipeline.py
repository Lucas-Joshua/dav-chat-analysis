"""Pipeline orchestration from raw input to processed outputs and charts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

from src import config
from src.feature_pipeline import FeaturePipeline
from src.modules.preprocessor import ChatPreprocessor
from src.visualizations import run_selected

DEFAULT_VISUALIZATION_SELECTIONS: dict[str, bool] = {
    "overall_emoji_distribution": True,
    "emoji_heatmap": False,   # removed from output per user request
    "chat_activity_by_hour": True,
    "chat_activity_weekday_weekend": True,
    "time_series_autocorrelation": True,
    "poisson_model": True,
    "incident_discussion_timeline": True,
    "incident_activity_correlation": True,
    "incident_context_projection": True,
    "incident_context_comparison": True,
    "incident_context_umap_analysis": True,
}

VISUALIZATION_FEATURES: dict[str, set[str]] = {
    "overall_emoji_distribution": {"emoji_category"},
    "emoji_heatmap": {"emoji_features"},
    "chat_activity_by_hour": {"time_features"},
    "chat_activity_weekday_weekend": {"time_features"},
    "time_series_autocorrelation": {"time_features"},
    "poisson_model": {"time_features"},
    "incident_discussion_timeline": {"incident_bow_features", "time_features"},
    "incident_activity_correlation": {"incident_bow_features", "time_features"},
    "incident_context_projection": {"incident_bow_features"},
    "incident_context_comparison": {"incident_bow_features"},
    "incident_context_umap_analysis": {"incident_bow_features"},
}


class PipelineRunner:
    """Run the end-to-end data pipeline with configurable visualization steps.

    :ivar logger: Module logger used for pipeline progress and failures.
    :vartype logger: logging.Logger
    :ivar visualization_selections: Mapping with visualization enable/disable flags.
    :vartype visualization_selections: dict[str, bool]
    :ivar preprocessor: Preprocessing component for loading, cleaning, and anonymizing.
    :vartype preprocessor: ChatPreprocessor
    """

    def __init__(
        self,
        visualization_selections: Mapping[str, bool] | None = None,
        preprocessor: ChatPreprocessor | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        """Initialize runner dependencies and runtime configuration.

        :param visualization_selections: Optional visualization selection overrides.
        :type visualization_selections: Mapping[str, bool] | None
        :param preprocessor: Optional preprocessor implementation.
        :type preprocessor: ChatPreprocessor | None
        :param feature_pipeline: Optional feature pipeline implementation.
        :type feature_pipeline: FeaturePipeline | None
        :return: None.
        :rtype: None
        """
        self.logger = logging.getLogger(__name__)
        self.visualization_selections = dict(
            visualization_selections or DEFAULT_VISUALIZATION_SELECTIONS
        )
        self.preprocessor = preprocessor or ChatPreprocessor()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply only feature steps required by enabled visualizations.

        :param df: Preprocessed chat dataframe.
        :type df: pd.DataFrame
        :return: Feature-enriched dataframe.
        :rtype: pd.DataFrame
        """
        requested_features = self._required_features()
        operations = self.feature_pipeline.resolve(requested_features)
        if not operations:
            self.logger.info("Step 4/6: skip feature engineering (not required)")
            return df

        self.logger.info(
            "Step 4/6: apply required feature operations: %s",
            ", ".join(operations),
        )
        df = self.feature_pipeline.apply(df, operations)
        self.logger.info("Dataset shape after processing: %s", df.shape)
        return df

    def _required_features(self) -> list[str]:
        """Return the features needed by the currently enabled visualizations."""
        required: set[str] = set()
        for name, enabled in self.visualization_selections.items():
            if enabled:
                required.update(VISUALIZATION_FEATURES.get(name, set()))
        return sorted(required)

    def _save_processed_files(self, df: pd.DataFrame) -> None:
        """Save the processed dataframe to parquet and CSV output files.

        :param df: Processed dataframe to persist.
        :type df: pd.DataFrame
        :return: None.
        :rtype: None
        """
        self.logger.info("Step 5/6: save processed files")
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.CLEAN_CHAT_PARQUET, index=False)
        df.to_csv(config.CLEAN_CHAT_CSV, index=False)

    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """Render enabled visualizations to the output image directory.

        :param df: Processed dataframe used as visualization input.
        :type df: pd.DataFrame
        :return: None.
        :rtype: None
        """
        self.logger.info("Step 6/6: generate visualizations")
        config.IMG_DIR.mkdir(parents=True, exist_ok=True)
        run_selected(df, self.visualization_selections, out_dir=config.IMG_DIR)

    def run(self, raw_path: str | Path) -> None:
        """Execute the full pipeline from raw input to persisted outputs.

        :param raw_path: Path to the raw chat export file.
        :type raw_path: str | Path
        :return: None.
        :rtype: None
        """
        enabled = [name for name, on in self.visualization_selections.items() if on]
        self.logger.info(
            "Pipeline config loaded. %d visualizations enabled.", len(enabled)
        )
        df = self.preprocessor.run(raw_path)
        df = self._add_features(df)
        self._save_processed_files(df)
        self._generate_visualizations(df)


def run_pipeline(raw_path: str | Path) -> None:
    """Run the default end-to-end data pipeline.

    :param raw_path: Path to the raw chat export file.
    :type raw_path: str | Path
    :return: None.
    :rtype: None
    """
    PipelineRunner().run(raw_path)
