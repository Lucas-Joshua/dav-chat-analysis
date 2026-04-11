"""Pipeline orchestration from raw input to processed outputs and charts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

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
from src.modules.author_stylometry import build_author_corpus, compute_stylometric_embedding
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
    "incident_discussion_timeline": False,
    "incident_activity_correlation": False,
    "author_clustering": False,
    "author_clustering_pca": False,
    "author_clustering_umap": False,
    "author_clustering_comparison": True,
    "umap_parameter_comparison": False,
    "time_series_activity": False,
    "time_series_autocorrelation": False,
    "poisson_model": False,
}

VISUALIZATION_REQUIREMENTS: dict[str, set[str]] = {
    "overall_emoji_distribution": {"emoji_category"},
    "emoji_heatmap": {"emoji_features"},
    "emoji_type_per_user": {"emoji_category"},
    "emoji_usage_by_hour": {"emoji_features", "time_features"},
    "negative_reaction_concentration": {"emoji_category"},
    "negative_reaction_diagnostic": {"emoji_category"},
    "negative_reaction_scatter": {"emoji_category"},
    "chat_activity_by_hour": {"time_features"},
    "chat_activity_distribution": {"time_features"},
    "response_time_suite": set(),
    "incident_discussion_timeline": set(),
    "incident_activity_correlation": set(),
    "author_clustering": {"stylometric_embedding"},
    "author_clustering_pca": {"stylometric_embedding"},
    "author_clustering_umap": {"stylometric_embedding"},
    "author_clustering_comparison": {"stylometric_embedding"},
    "umap_parameter_comparison": {"stylometric_embedding"},
    "time_series_activity": {"time_features"},
    "time_series_autocorrelation": {"time_features"},
    "poisson_model": {"time_features"},
}

FEATURE_DEPENDENCIES: dict[str, set[str]] = {
    "emoji_features": set(),
    "emoji_category": {"emoji_features"},
    "time_features": set(),
    "message_length": set(),
    "has_emoji_feature": {"emoji_features"},
    "incident_bow_features": set(),
    "stylometric_embedding": set(),
}

FEATURE_OPERATIONS = {
    "emoji_features": add_emoji_features,
    "emoji_category": add_emoji_category,
    "time_features": add_time_features,
    "message_length": add_message_length,
    "has_emoji_feature": add_has_emoji_feature,
    "incident_bow_features": add_incident_bow_features,
}

FEATURE_EXECUTION_ORDER = [
    "emoji_features",
    "emoji_category",
    "time_features",
    "message_length",
    "has_emoji_feature",
    "incident_bow_features",
    "stylometric_embedding",
]


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
    ) -> None:
        """Initialize runner dependencies and runtime configuration.

        :param visualization_selections: Optional visualization selection overrides.
        :type visualization_selections: Mapping[str, bool] | None
        :param preprocessor: Optional preprocessor implementation.
        :type preprocessor: ChatPreprocessor | None
        :return: None.
        :rtype: None
        """
        self.logger = logging.getLogger(__name__)
        self.visualization_selections = dict(
            visualization_selections or VISUALIZATION_SELECTIONS
        )
        self.preprocessor = preprocessor or ChatPreprocessor()

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply only feature steps required by enabled visualizations.

        :param df: Preprocessed chat dataframe.
        :type df: pd.DataFrame
        :return: Feature-enriched dataframe.
        :rtype: pd.DataFrame
        """
        required_operations = self._resolve_required_operations()
        if not required_operations:
            self.logger.info("Step 4/6: skip feature engineering (not required)")
            return df

        self.logger.info(
            "Step 4/6: apply required feature operations: %s",
            ", ".join(required_operations),
        )
        for operation in required_operations:
            if operation == "stylometric_embedding":
                df = self._add_stylometric_embedding(df)
                continue
            transform = FEATURE_OPERATIONS[operation]
            df = transform(df)
        self.logger.info("Dataset shape after processing: %s", df.shape)
        return df

    def _resolve_required_operations(self) -> list[str]:
        """Resolve required feature operations including dependencies.

        :return: Ordered list of required feature-operation identifiers.
        :rtype: list[str]
        """
        enabled_visualizations = {
            name for name, enabled in self.visualization_selections.items() if enabled
        }
        required: set[str] = set()
        for visualization_name in enabled_visualizations:
            required.update(VISUALIZATION_REQUIREMENTS.get(visualization_name, set()))

        resolved: set[str] = set()
        stack = list(required)
        while stack:
            operation = stack.pop()
            if operation in resolved:
                continue
            resolved.add(operation)
            stack.extend(FEATURE_DEPENDENCIES.get(operation, set()))

        return [op for op in FEATURE_EXECUTION_ORDER if op in resolved]

    def _add_stylometric_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trigram-based stylometric embeddings and store them in ``df.attrs``.

        :param df: Feature-engineered chat dataframe.
        :type df: pd.DataFrame
        :return: Input dataframe with stylometry attributes attached.
        :rtype: pd.DataFrame
        """
        self.logger.info("Step 4b/6: compute stylometric embeddings (trigrams → distance → PCA/tSNE)")
        try:
            chunk_size = 500
            texts, labels = build_author_corpus(df, n=chunk_size, min_parts=2, author_col="sender")
            df.attrs["stylometry_chunk_size"] = chunk_size
            df.attrs["stylometry_texts"] = texts
            df.attrs["stylometry_labels"] = labels
            tsne_emb = compute_stylometric_embedding(texts, method="tSNE")
            pca_emb  = compute_stylometric_embedding(texts, method="PCA")
            umap_emb = compute_stylometric_embedding(texts, method="UMAP")
            df.attrs["stylometry_tsne"] = tsne_emb
            df.attrs["stylometry_pca"]  = pca_emb
            df.attrs["stylometry_umap"] = umap_emb

            # K-means clusters on the t-SNE 2-D embedding (first 2 components).
            n_clusters = 4
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            xy = np.asarray(tsne_emb)[:, :2]
            chunk_cluster_labels = km.fit_predict(xy).tolist()
            sil_score = silhouette_score(xy, chunk_cluster_labels)
            self.logger.info("KMeans silhouette score (tSNE 2D, k=%d): %.4f", n_clusters, sil_score)
            # Aggregate: majority-vote cluster per author.
            _tmp = pd.DataFrame({"author": labels, "cluster": chunk_cluster_labels})
            author_cluster = (
                _tmp.groupby("author")["cluster"]
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
            )
            df.attrs["stylometry_author_cluster"] = author_cluster
            df.attrs["stylometry_n_clusters"]     = n_clusters
            self.logger.info("Stylometric embeddings computed for %d chunks", len(texts))
        except Exception:
            self.logger.exception("Stylometric embedding failed — author clustering will be skipped")
        return df

    def _save_processed_files(self, df: pd.DataFrame) -> None:
        """Save the processed dataframe to parquet and CSV output files.

        :param df: Processed dataframe to persist.
        :type df: pd.DataFrame
        :return: None.
        :rtype: None
        """
        self.logger.info("Step 5/6: save processed files")
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Parquet serialises df.attrs as JSON; strip numpy arrays first so they
        # don't cause a TypeError.  The original df (with attrs) is kept intact
        # for the visualisation step that follows.
        df_save = df.copy()
        df_save.attrs = {k: v for k, v in df.attrs.items() if not isinstance(v, np.ndarray)}
        df_save.to_parquet(config.PROCESSED_DIR / "clean_chat_processed.parquet", index=False)
        df_save.to_csv(config.PROCESSED_DIR / "clean_chat_processed.csv", index=False)

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
