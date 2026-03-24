"""Low-level Plotly scatterplot for author stylometric clustering.

Gestalt principles applied
--------------------------
* Common Region  – a confidence ellipse encloses every author's chunks,
                   making the group immediately readable without needing colour.
* Figure / Ground – 4 K-means clusters each get a distinct colour; all points
                    belonging to a cluster share that colour, so the groups
                    pop out of the background immediately.
* Proximity       – real author names are placed directly at each cluster's
                    centroid, removing indirection of a separate legend.
* Similarity      – all chunks from authors in the same cluster share one hue.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS

logger = logging.getLogger(__name__)

# Colour palette — one vivid colour per cluster (up to 6 clusters supported).
_CLUSTER_COLORS = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#8338EC", "#FB5607"]

# Path to the real-name mapping (relative to project root).
_MAPPING_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "user_mapping.csv"

# Optional cluster-age anchors (1-indexed cluster labels in plot/output).
_CLUSTER_AGE_HINTS = {
    1: "jong-midden mix (ankers: Lucas 20-35, Suzanne eind 30)",
    2: "jongere mix (anker: Harmen eind 20)",
    3: "oudere groep (anker: Rene 50+)",
    4: "gemengde mix (ankers: Sander/Sabien/Esther)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_name_mapping() -> dict[str, str]:
    """Return {pseudo_name: real_name} from user_mapping.csv, if it exists."""
    mapping: dict[str, str] = {}
    if not _MAPPING_PATH.exists():
        return mapping
    with _MAPPING_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            real = row.get("real_name", "").strip().lstrip("~").strip()
            pseudo = row.get("pseudo_name", "").strip()
            if real and pseudo:
                mapping[pseudo] = real
    return mapping


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    n_std: float = 1.5,
    n_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_ell, y_ell) for an n_std-sigma confidence ellipse (numpy only)."""
    if len(x) < 2:
        t = np.linspace(0, 2 * np.pi, n_points)
        return x[0] + 0.5 * np.cos(t), y[0] + 0.5 * np.sin(t)

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    a = n_std * np.sqrt(max(eigenvalues[0], 1e-9))
    b = n_std * np.sqrt(max(eigenvalues[1], 1e-9))

    t = np.linspace(0, 2 * np.pi, n_points)
    ell_x = a * np.cos(t)
    ell_y = b * np.sin(t)
    v = eigenvectors
    return (
        v[0, 0] * ell_x + v[0, 1] * ell_y + mean_x,
        v[1, 0] * ell_x + v[1, 1] * ell_y + mean_y,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_author_clustering(
    df: pd.DataFrame,
    out_path: str | Path = "img/author_clustering.png",
    method: Literal["PCA", "tSNE", "UMAP"] = "tSNE",
    **kwargs,
) -> None:
    """Render a 2-D scatter coloured by K-means writing-style cluster.

    Each point is a 500-char chunk of one author's messages.  Authors are
    coloured by their majority K-means cluster, and confidence ellipses
    (Gestalt: Common Region) enclose every author's chunks.  Real names are
    shown directly on the plot (Gestalt: Proximity) so the viewer can
    immediately relate writing-style groups to known people.

    Parameters
    ----------
    df:
        Processed chat DataFrame with stylometry data in ``df.attrs``.
    out_path:
        Destination path for the exported PNG.
    method:
        Which pre-computed embedding to use: ``"PCA"``, ``"tSNE"``, or ``"UMAP"``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attr_key = {
        "tSNE": "stylometry_tsne",
        "PCA": "stylometry_pca",
        "UMAP": "stylometry_umap",
    }[method]
    embedding: np.ndarray | None = df.attrs.get(attr_key)
    labels: list[str] | None = df.attrs.get("stylometry_labels")
    author_cluster: dict[str, int] | None = df.attrs.get("stylometry_author_cluster")
    n_clusters: int = df.attrs.get("stylometry_n_clusters", 4)

    if embedding is None or labels is None:
        raise RuntimeError(
            "Stylometric embedding not found in df.attrs. "
            "Ensure _add_stylometric_embedding ran successfully in the pipeline."
        )

    # Use only first 2 dimensions.
    xy = np.asarray(embedding)[:, :2]
    label_arr = np.asarray(labels)

    plot_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "author": label_arr})

    # Only authors with ≥ 3 chunks get an ellipse.
    counts = plot_df["author"].value_counts()
    valid_authors: list[str] = counts[counts >= 3].index.tolist()
    plot_df = plot_df[plot_df["author"].isin(valid_authors)].reset_index(drop=True)

    # Assign cluster colour to each chunk row.
    if author_cluster:
        plot_df["cluster"] = plot_df["author"].map(
            lambda a: author_cluster.get(a, 0)
        )
    else:
        plot_df["cluster"] = 0

    # Real-name lookup.
    name_map = _load_name_mapping()
    plot_df["display_name"] = plot_df["author"].map(
        lambda a: name_map.get(a, a)
    )

    n_authors = len(valid_authors)
    logger.info("Plotting %d authors (%d chunks), %d clusters", n_authors, len(plot_df), n_clusters)

    # ── Build figure (storyboard: one panel per cluster) ─────────────────
    cols = 2 if n_clusters <= 4 else 3
    rows = int(math.ceil(n_clusters / cols))

    subplot_titles: list[str] = []
    for cluster_id in range(n_clusters):
        cluster_num = cluster_id + 1
        subplot_titles.append(f"C{cluster_num} · {_CLUSTER_AGE_HINTS.get(cluster_num, 'onbekend')}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    x_min, x_max = float(plot_df["x"].min()), float(plot_df["x"].max())
    y_min, y_max = float(plot_df["y"].min()), float(plot_df["y"].max())
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    for cluster_id in range(n_clusters):
        r = cluster_id // cols + 1
        c = cluster_id % cols + 1
        hex_col = _CLUSTER_COLORS[cluster_id % len(_CLUSTER_COLORS)]
        subset = plot_df[plot_df["cluster"] == cluster_id]

        # Context layer: all points in soft gray.
        fig.add_trace(
            go.Scatter(
                x=plot_df["x"],
                y=plot_df["y"],
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                marker=dict(color="rgba(120,120,120,0.20)", size=5),
            ),
            row=r,
            col=c,
        )

        # Cluster zone.
        pts = subset[["x", "y"]].to_numpy()
        if len(pts) >= 3:
            ex, ey = _confidence_ellipse(pts[:, 0], pts[:, 1], n_std=1.8)
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    fill="toself",
                    fillcolor=_hex_to_rgba(hex_col, 0.14),
                    line=dict(color=_hex_to_rgba(hex_col, 0.6), width=2),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r,
                col=c,
            )

        # Cluster points.
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=hex_col,
                    size=7,
                    opacity=0.90,
                    line=dict(width=0.4, color="white"),
                ),
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=subset["display_name"],
            ),
            row=r,
            col=c,
        )

        # Cluster centroid.
        if not subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=[float(subset["x"].mean())],
                    y=[float(subset["y"].mean())],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="skip",
                    marker=dict(
                        size=22,
                        color=hex_col,
                        symbol="x",
                        line=dict(width=2.2, color="white"),
                    ),
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="",
            range=x_range,
            row=r,
            col=c,
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="",
            range=y_range,
            row=r,
            col=c,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    chunk_size = df.attrs.get("stylometry_chunk_size", 500)
    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 40, "r": 40, "t": 120, "b": 50},
        )
    )
    fig.update_layout(
        title={
            "text": (
                f"Schrijfstijl-clusters Storyboard · {method}"
                f"<br><sup>Karakter-trigrammen · Chunk {chunk_size} tekens · Cosine-afstand · {n_clusters} clusters (K-means)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=False,
        height=max(900, rows * 520),
        width=1500 if cols == 2 else 1800,
    )
    fig.update_annotations(font=dict(size=12))

    fig.write_image(str(out_path), scale=2)
    logger.info("Author clustering plot saved to %s", out_path)


def plot_author_reduction_comparison(
    df: pd.DataFrame,
    out_path: str | Path = "img/author_clustering_comparison.png",
) -> None:
    """Render side-by-side PCA, t-SNE, and UMAP comparison with shared axes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] | None = df.attrs.get("stylometry_labels")
    tsne_emb: np.ndarray | None = df.attrs.get("stylometry_tsne")
    pca_emb: np.ndarray | None = df.attrs.get("stylometry_pca")
    umap_emb: np.ndarray | None = df.attrs.get("stylometry_umap")
    author_cluster: dict[str, int] | None = df.attrs.get("stylometry_author_cluster")
    n_clusters: int = df.attrs.get("stylometry_n_clusters", 4)
    if labels is None or tsne_emb is None or pca_emb is None or umap_emb is None:
        raise RuntimeError("Required embeddings (PCA/tSNE/UMAP) not found in df.attrs.")

    label_arr = np.asarray(labels)
    df_list: list[pd.DataFrame] = []
    method_specs = [("t-SNE", tsne_emb), ("UMAP", umap_emb), ("PCA", pca_emb)]
    for method_name, emb in method_specs:
        tmp = pd.DataFrame({
            "x": np.asarray(emb)[:, 0],
            "y": np.asarray(emb)[:, 1],
            "author": label_arr,
            "method": method_name,
        })
        df_list.append(tmp)
    all_df = pd.concat(df_list, ignore_index=True)

    counts = all_df[all_df["method"] == "t-SNE"]["author"].value_counts()
    valid_authors = counts[counts >= 3].index.tolist()
    all_df = all_df[all_df["author"].isin(valid_authors)].copy()
    if author_cluster:
        all_df["cluster"] = all_df["author"].map(lambda a: author_cluster.get(a, 0))
    else:
        all_df["cluster"] = 0

    x_min, x_max = float(all_df["x"].min()), float(all_df["x"].max())
    y_min, y_max = float(all_df["y"].min()), float(all_df["y"].max())
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_span = x_max - x_min
    y_span = y_max - y_min
    half_span = max(x_span, y_span) * 0.53 if max(x_span, y_span) > 0 else 1.0
    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    for ax, method_name in zip(axes, ["t-SNE", "UMAP", "PCA"]):
        subset = all_df[all_df["method"] == method_name]
        ax.scatter(
            subset["x"],
            subset["y"],
            c="lightgray",
            alpha=0.15,
            s=20,
            linewidths=0,
            zorder=1,
        )
        for cluster_id in range(n_clusters):
            csub = subset[subset["cluster"] == cluster_id]
            if csub.empty:
                continue
            color = _CLUSTER_COLORS[cluster_id % len(_CLUSTER_COLORS)]
            ax.scatter(
                csub["x"],
                csub["y"],
                c=color,
                alpha=0.9,
                s=26,
                linewidths=0,
                zorder=2,
            )
            ax.scatter(
                [csub["x"].mean()],
                [csub["y"].mean()],
                c=color,
                marker="X",
                s=180,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
        ax.set_title(method_name, pad=4, fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.25)

    fig.suptitle("Writing-Style Clustering Comparison", y=0.93, fontsize=13)
    plt.subplots_adjust(wspace=0.15, hspace=0.15, bottom=0.16, top=0.88, left=0.04, right=0.99)
    caption = (
        "Elke stip stelt een tekstfragment voor; nabijheid betekent vergelijkbare schrijfstijl. "
        "PCA toont globale structuur, t-SNE lokale verschillen en UMAP zit daartussen. "
        "De overlap suggereert dat schrijfstijl een continu spectrum vormt. "
        "Zelfs na tuning blijft de data geconcentreerd. "
        "Alleen onderlinge afstanden zijn betekenisvol."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=10, wrap=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    logger.info("Author reduction comparison plot saved to %s", out_path)
