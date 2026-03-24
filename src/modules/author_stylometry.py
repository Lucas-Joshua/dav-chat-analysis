"""Stylometric analysis: chunk text per author, vectorize with char-trigrams, reduce to 3D.

Implements the approach from MADS-DAV notebook 06.2-modelling.ipynb (cells 40–41):
  1. Concatenate each author's messages and split into fixed-size character chunks.
  2. Represent each chunk as a character-level trigram count vector (CountVectorizer).
  3. Compute pairwise cosine distances between all chunk vectors.
  4. Reduce to 3D with PCA or t-SNE for visualisation.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")
_MULTI_SPACE = re.compile(r" +")


def _clean_chunk(text: str) -> str:
    """Remove URLs and collapse multiple spaces in a single chunk."""
    text = _URL_PATTERN.sub("", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def build_author_corpus(
    df: pd.DataFrame,
    n: int = 500,
    min_parts: int = 2,
    author_col: str = "author",
    message_col: str = "message",
) -> tuple[list[str], list[str]]:
    """Split each author's concatenated messages into n-char chunks.

    Parameters
    ----------
    df:
        DataFrame containing at least ``author_col`` and ``message_col``.
    n:
        Chunk size in characters. Experiment with 200, 500, or 1000.
    min_parts:
        Minimum number of chunks an author must produce to be included.
        Authors with too little text are dropped.
    author_col:
        Name of the column containing author identifiers.
    message_col:
        Name of the column containing message text.

    Returns
    -------
    texts:
        Flat list of cleaned text chunks across all authors.
    labels:
        Corresponding author label for every chunk in ``texts``.
    """
    authors = df[author_col].unique()
    corpus: dict[str, list[str]] = {}

    for author in authors:
        subset = df[df[author_col] == author]
        longseq = " ".join(subset[message_col].astype(str))
        parts = [longseq[i : i + n] for i in range(0, len(longseq), n)]
        parts = [_clean_chunk(chunk) for chunk in parts]
        # Drop the last (partial) chunk and very short chunks
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
    random_state: int = 42,
) -> np.ndarray:
    """Compute a 2-D stylometric embedding for a list of text chunks.

    Steps
    -----
    1. Fit a character-level trigram ``CountVectorizer`` on all chunks.
    2. Compute the pairwise cosine distance matrix.
    3. Reduce the distance matrix to 2 dimensions with PCA or t-SNE.

    Parameters
    ----------
    texts:
        List of text chunks (output of :func:`build_author_corpus`).
    method:
        Dimensionality-reduction method: ``"PCA"``, ``"tSNE"``, or ``"UMAP"``.
    random_state:
        Random seed for reproducible t-SNE results.

    Returns
    -------
    np.ndarray of shape (len(texts), 3)
    """
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    X = np.asarray(vectorizer.fit_transform(texts).todense())
    logger.info("Trigram matrix shape: %s", X.shape)

    dist = cosine_distances(X, X)

    if method == "PCA":
        reducer = PCA(n_components=3, random_state=random_state)
        embedding = reducer.fit_transform(dist)
    else:
        # Optional denoising before non-linear methods.
        pca_dims = min(50, X.shape[1], len(texts))
        X_input = X
        if pca_dims >= 3 and pca_dims < X.shape[1]:
            X_input = PCA(n_components=pca_dims, random_state=random_state).fit_transform(X)

        if method == "tSNE":
            perplexity = 30 if len(texts) > 30 else max(5, len(texts) // 3)
            reducer = TSNE(
                n_components=3,
                metric="cosine",
                init="random",
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
            )
            embedding = reducer.fit_transform(X_input)
            logger.info("tSNE configured with perplexity=%d, random_state=%d", perplexity, random_state)
        else:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=5,
                min_dist=0.3,
                random_state=random_state,
                metric="cosine",
            )
            embedding = reducer.fit_transform(X_input)
            logger.info("UMAP configured with n_neighbors=15, min_dist=0.1, random_state=%d", random_state)
    logger.info("Embedding computed with method=%s, shape=%s", method, embedding.shape)
    return embedding
