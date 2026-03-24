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
        df = self._add_stylometric_embedding(df)
        return df

    def _add_stylometric_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trigram-based stylometric embeddings and store in df.attrs."""
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
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            import numpy as np
            n_clusters = 4
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            xy = np.asarray(tsne_emb)[:, :2]
            chunk_cluster_labels = km.fit_predict(xy).tolist()
            sil_score = silhouette_score(xy, chunk_cluster_labels)
            self.logger.info("KMeans silhouette score (tSNE 2D, k=%d): %.4f", n_clusters, sil_score)
            # Aggregate: majority-vote cluster per author.
            import pandas as _pd
            _tmp = _pd.DataFrame({"author": labels, "cluster": chunk_cluster_labels})
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
        """Save processed dataframe to parquet and CSV outputs."""
        self.logger.info("Step 5/6: save processed files")
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Parquet serialises df.attrs as JSON; strip numpy arrays first so they
        # don't cause a TypeError.  The original df (with attrs) is kept intact
        # for the visualisation step that follows.
        import numpy as np
        df_save = df.copy()
        df_save.attrs = {k: v for k, v in df.attrs.items() if not isinstance(v, np.ndarray)}
        df_save.to_parquet(config.PROCESSED_DIR / "clean_chat_processed.parquet", index=False)
        df_save.to_csv(config.PROCESSED_DIR / "clean_chat_processed.csv", index=False)

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
