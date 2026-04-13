"""Stylometric analysis: chunk text per author, vectorize with char-trigrams, reduce to 3D.

Implements the approach from MADS-DAV notebook 06.2-modelling.ipynb (cells 40–41):
  1. Concatenate each author's messages and split into fixed-size character chunks.
  2. Represent each chunk as a character-level trigram count vector (CountVectorizer).
  3. Compute pairwise Manhattan distances between all chunk vectors.
  4. Reduce to 3D with PCA or t-SNE for visualisation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import manhattan_distances

try:
    import umap  # type: ignore[import-not-found]
except ImportError:
    umap = None

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")
_MULTI_SPACE = re.compile(r" +")


def _clean_chunk(text: str) -> str:
    """Remove URLs and collapse multiple spaces in a single chunk.

    :param text: Input text chunk.
    :type text: str
    :return: Normalized text chunk.
    :rtype: str
    """
    text = _URL_PATTERN.sub("", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def build_author_corpus(
    df: pd.DataFrame,
    n: int = 500,
    min_parts: int = 2,
    author_col: str = "author",
    message_col: str = "message",
) -> tuple[list[str], list[str]]:
    """Split each author's concatenated messages into fixed-size chunks.

    :param df: Dataframe containing author and message columns.
    :type df: pd.DataFrame
    :param n: Chunk size in characters.
    :type n: int
    :param min_parts: Minimum chunks required per author.
    :type min_parts: int
    :param author_col: Name of the author column.
    :type author_col: str
    :param message_col: Name of the message-text column.
    :type message_col: str
    :return: Tuple with chunk texts and parallel author labels.
    :rtype: tuple[list[str], list[str]]
    """
    authors = df[author_col].unique()
    corpus: dict[str, list[str]] = {}

    for author in authors:
        subset = df[df[author_col] == author]
        longseq = " ".join(subset[message_col].astype(str))
        parts = [longseq[i : i + n] for i in range(0, len(longseq), n)]
        parts = [_clean_chunk(chunk) for chunk in parts]
        parts = [p for p in parts[:-1] if len(p) >= n // 2]
        if len(parts) >= min_parts:
            corpus[author] = parts

    logger.info(
        "Corpus built: %d authors, %d chunks total (n=%d, min_parts=%d)",
        len(corpus),
        sum(len(v) for v in corpus.values()),
        n,
        min_parts,
    )

    texts = [chunk for chunks in corpus.values() for chunk in chunks]
    labels = [author for author, chunks in corpus.items() for _ in chunks]
    return texts, labels


def compute_stylometric_embedding(
    texts: list[str],
    method: Literal["PCA", "tSNE", "UMAP"] = "tSNE",
    n_components: int = 3,
    random_state: int = 42,
    pre_pca_dims: int | None = 50,
    **reducer_kwargs: Any,
) -> np.ndarray:
    """Compute a stylometric embedding for a list of text chunks.

    :param texts: Text chunks produced by ``build_author_corpus``.
    :type texts: list[str]
    :param method: Dimensionality-reduction method (``PCA``, ``tSNE``, or ``UMAP``).
    :type method: Literal["PCA", "tSNE", "UMAP"]
    :param n_components: Number of dimensions in the output embedding.
    :type n_components: int
    :param random_state: Random seed for reproducible embeddings.
    :type random_state: int
    :param pre_pca_dims: Optional intermediate PCA dimensions for non-linear reducers.
    :type pre_pca_dims: int | None
    :param reducer_kwargs: Extra keyword arguments forwarded to the selected reducer.
    :type reducer_kwargs: Any
    :return: Embedding array with one row per input chunk.
    :rtype: np.ndarray
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    method_map = {
        "PCA": "PCA",
        "TSNE": "tSNE",
        "T-SNE": "tSNE",
        "UMAP": "UMAP",
    }
    method_key = method_map.get(str(method).strip().upper())
    if method_key is None:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of: PCA, tSNE, UMAP."
        )

    for reserved_key in ("n_components", "random_state"):
        if reserved_key in reducer_kwargs:
            raise ValueError(
                f"Do not pass '{reserved_key}' via reducer_kwargs; use the named function parameter instead."
            )

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    X = np.asarray(vectorizer.fit_transform(texts).todense())
    logger.info("Trigram matrix shape: %s", X.shape)

    dist = manhattan_distances(X, X)

    if method_key == "PCA":
        reducer = PCA(
            n_components=n_components,
            **reducer_kwargs,
        )
        embedding = reducer.fit_transform(dist)

    elif method_key in {"tSNE", "UMAP"}:
        X_input = dist
        if pre_pca_dims is not None:
            pca_dims = min(pre_pca_dims, X_input.shape[1], len(texts))
            if pca_dims >= n_components and pca_dims < X_input.shape[1]:
                X_input = PCA(n_components=pca_dims).fit_transform(X_input)

        if method_key == "tSNE":
            max_perplexity = max(1, len(texts) - 1)
            default_perplexity = 30 if len(texts) > 30 else max(5, len(texts) // 3)
            perplexity = min(default_perplexity, max_perplexity)
            reducer = TSNE(
                n_components=n_components,
                metric="euclidean",
                init="random",
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
                **reducer_kwargs,
            )
            embedding = reducer.fit_transform(X_input)
            logger.info("tSNE configured with perplexity=%d, random_state=%d", perplexity, random_state)

        elif method_key == "UMAP":
            if umap is None:
                raise ImportError(
                    "UMAP was requested but the 'umap-learn' package is not installed."
                )
            umap_kwargs = {
                "n_neighbors": 5,
                "min_dist": 0.3,
                "metric": "euclidean",
            }
            umap_kwargs.update(reducer_kwargs)
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                **umap_kwargs,
            )
            embedding = reducer.fit_transform(X_input)
            logger.info(
                "UMAP configured with n_neighbors=%d, min_dist=%.2f, random_state=%d",
                reducer.n_neighbors,
                reducer.min_dist,
                random_state,
            )

    else:
        raise RuntimeError(f"Unhandled method resolution: {method_key}")

    logger.info("Embedding computed with method=%s, shape=%s", method_key, embedding.shape)
    return embedding
