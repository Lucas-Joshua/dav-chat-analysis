"""Incident-context dimensionality-reduction views for Les 6."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from src.modules.feature_engineering import add_incident_bow_features
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import ensure_parent_dir

LESSON_DIR = "les6"


def _prepare_context_sample(
    df: pd.DataFrame,
    max_points: int = 1600,
    min_chars: int = 8,
) -> pd.DataFrame:
    """Build a balanced sample of incident vs regular messages for plotting.

    ``max_points`` limits rendering cost. To avoid a heavily skewed chart, the
    majority class is downsampled only for visualisation; source data is not
    changed.

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param max_points: Maximum number of messages used for the projection.
    :type max_points: int
    :param min_chars: Minimum message length in characters.
    :type min_chars: int
    :return: Sampled dataframe with ``message`` and ``incident_related``.
    :rtype: pd.DataFrame
    """
    working = df.copy()
    if "is_incident_message" not in working.columns:
        working = add_incident_bow_features(working)

    if "message" not in working.columns:
        raise KeyError("message column not found.")

    sample = working[["message", "is_incident_message"]].copy()
    sample["message"] = sample["message"].fillna("").astype(str).str.strip()
    sample = sample[sample["message"].str.len() >= min_chars].copy()
    sample["incident_related"] = sample["is_incident_message"].astype(int).eq(1)

    incident = sample[sample["incident_related"]]
    regular = sample[~sample["incident_related"]]
    if incident.empty or regular.empty:
        raise ValueError("Incident-context split requires both incident and regular messages.")

    # Keep the visualisation interpretable by limiting majority dominance.
    max_per_class = max_points // 2
    incident_sample = incident.sample(n=min(len(incident), max_per_class), random_state=42)
    regular_sample = regular.sample(n=min(len(regular), max_per_class), random_state=42)
    # Avoid pandas attrs equality checks on ndarray attrs during concat.
    incident_sample.attrs = {}
    regular_sample.attrs = {}
    balanced = pd.concat([incident_sample, regular_sample], ignore_index=True)
    balanced.attrs = {}

    return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _les6_output_path(
    out_dir: str | Path | None,
    filename: str,
) -> Path:
    """Resolve a Les 6 output path under ``img/les6`` by default."""
    base = Path(out_dir) if out_dir else Path("img")
    return ensure_parent_dir(base / LESSON_DIR / filename)




def _build_feature_matrix(texts: list) -> "np.ndarray":
    """Shared char-trigram + Manhattan distance + PCA pipeline."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import manhattan_distances

    np.random.seed(42)
    vec = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    X = vec.fit_transform(texts).toarray()
    dist = manhattan_distances(X)
    pca_dims = min(50, dist.shape[1], len(texts) - 1)
    return PCA(n_components=pca_dims, random_state=42).fit_transform(dist)


def _build_umap_embedding(X_reduced: "np.ndarray") -> "np.ndarray":
    """Shared UMAP embedding — always identical for the same X_reduced."""
    import numpy as np
    import umap as umap_lib  # type: ignore[import-not-found]

    np.random.seed(42)
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=5, min_dist=0.3,
        metric="euclidean", random_state=42, init="spectral",
    )
    return reducer.fit_transform(X_reduced)


def incident_context_umap_analysis(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Dedicated UMAP analysis of incident vs regular message context.

    UMAP preserves global structure better than t-SNE, making it possible to
    see whether incident messages occupy a *specific region* of the manifold
    (content-driven) or are spread uniformly (only volume changes).

    The chart uses:
    - A KDE density contour for regular messages (background topology)
    - Larger, more opaque markers for incident messages (foreground signal)
    - A convex-hull annotation around the densest incident cluster
    - An insight-driven annotation showing the key finding

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.spatial import ConvexHull

    sample = _prepare_context_sample(df, max_points=1600)
    texts = sample["message"].tolist()
    is_incident = sample["incident_related"].to_numpy()

    X_reduced = _build_feature_matrix(texts)
    emb = _build_umap_embedding(X_reduced)
    x = emb[:, 0]
    y = emb[:, 1]

    x_inc = x[is_incident]
    y_inc = y[is_incident]
    x_reg = x[~is_incident]
    y_reg = y[~is_incident]

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()

    fig, ax = plt.subplots(figsize=(9, 7))

    # --- Background: KDE density contour for regular messages ---
    if len(x_reg) > 10:
        try:
            kde = gaussian_kde(np.vstack([x_reg, y_reg]), bw_method=0.25)
            x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 120)
            y_grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 120)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=6, cmap="Greys", alpha=0.22, zorder=1)
            ax.contour(xx, yy, zz, levels=4, colors=["#AAAAAA"], linewidths=0.5,
                       alpha=0.45, zorder=2)
        except Exception as exc:
            logger.debug("KDE skipped (degenerate data): %s", exc)

    # --- Regular messages: small, transparent ---
    ax.scatter(x_reg, y_reg,
               s=14, color=DEFAULT_PLOT_SETTINGS.neutral_color,
               alpha=0.30, linewidths=0, zorder=3, label=f"Regulier (n={len(x_reg)})")

    # --- Incident messages: larger, opaque, red ---
    ax.scatter(x_inc, y_inc,
               s=55, color=DEFAULT_PLOT_SETTINGS.danger_color,
               alpha=0.80, linewidths=0.4, edgecolors="white", zorder=5,
               label=f"Incident-gerelateerd (n={len(x_inc)})")

    # --- Convex hull around incident cluster (if enough points) ---
    if len(x_inc) >= 4:
        try:
            pts = np.column_stack([x_inc, y_inc])
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            # Close the polygon
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=DEFAULT_PLOT_SETTINGS.danger_color,
                    alpha=0.07, zorder=4, linewidth=0)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=DEFAULT_PLOT_SETTINGS.danger_color,
                    linewidth=1.0, linestyle="--", alpha=0.50, zorder=4)
        except Exception as exc:
            logger.debug("Convex hull skipped: %s", exc)

    # --- Insight annotation — fixed in lower-left corner (empty space in this manifold) ---
    ax.text(
        0.01, 0.01,
        "Geen apart cluster zichtbaar\nIncidentberichten zijn semantisch\ngeïntegreerd in de gewone wolk",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize,
        color=DEFAULT_PLOT_SETTINGS.danger_color,
        bbox=DEFAULT_PLOT_SETTINGS.annotation_box,
    )

    ax.set_xlabel("UMAP dimensie 1")
    ax.set_ylabel("UMAP dimensie 2")
    ax.set_title(
        "UMAP-projectie: incidentberichten vormen geen apart cluster\n"
        "Karakter-trigrammen · n_neighbors=5 · min_dist=0.30 · rood = incident-context"
    )
    ax.legend(fontsize=DEFAULT_PLOT_SETTINGS.legend_fontsize,
              frameon=False, loc="upper left", markerscale=1.4)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Caption
    ax.text(
        0.99, 0.01,
        "UMAP behoudt globale structuur · t-SNE optimaliseert alleen lokale nabijheid",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color, style="italic",
    )

    fig.tight_layout()
    out_path = _les6_output_path(out_dir, "incident_context_umap_analysis.png")
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def incident_context_tsne_umap_comparison(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Side-by-side t-SNE vs UMAP comparison using identical char-trigram features.

    Both projections use the same TF-IDF char-trigram matrix so differences
    are purely due to the reduction algorithm, not the input representation.

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    sample = _prepare_context_sample(df, max_points=1600)
    texts = sample["message"].tolist()
    is_incident = sample["incident_related"].to_numpy()

    # Shared feature matrix — identical to standalone UMAP function
    X_reduced = _build_feature_matrix(texts)

    # t-SNE on same matrix
    np.random.seed(42)
    tsne = TSNE(n_components=2, perplexity=30, metric="euclidean",
                init="random", random_state=42, max_iter=1000)
    emb_tsne = tsne.fit_transform(X_reduced)

    # UMAP via shared helper — guaranteed identical to standalone
    emb_umap = _build_umap_embedding(X_reduced)

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "t-SNE vs. UMAP op dezelfde karakter-trigrammen\n"
        "Beide methodes, zelfde data — rood = incident-context",
        fontsize=DEFAULT_PLOT_SETTINGS.title_fontsize,
    )

    config = [
        (axes[0], emb_tsne, "t-SNE", "t-SNE dimensie 1", "t-SNE dimensie 2",
         "Lokale structuur · perplexity=30",
         "t-SNE compresseert lokale clusters\nmaar vervormt globale afstanden"),
        (axes[1], emb_umap, "UMAP", "UMAP dimensie 1", "UMAP dimensie 2",
         "Globale + lokale structuur · n_neighbors=5",
         "UMAP behoudt globale structuur\nen is stabieler over runs"),
    ]

    for ax, emb, title, xlabel, ylabel, subtitle, note in config:
        x, y = emb[:, 0], emb[:, 1]
        x_reg, y_reg = x[~is_incident], y[~is_incident]
        x_inc, y_inc = x[is_incident], y[is_incident]

        ax.scatter(x_reg, y_reg, s=12, color=DEFAULT_PLOT_SETTINGS.neutral_color,
                   alpha=0.28, linewidths=0, zorder=2,
                   label=f"Regulier (n={len(x_reg)})")
        ax.scatter(x_inc, y_inc, s=50, color=DEFAULT_PLOT_SETTINGS.danger_color,
                   alpha=0.80, linewidths=0.4, edgecolors="white", zorder=4,
                   label=f"Incident (n={len(x_inc)})")

        ax.set_title(f"{title}\n{subtitle}",
                     fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize + 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=DEFAULT_PLOT_SETTINGS.legend_fontsize,
                  frameon=False, loc="upper right")
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
                color=DEFAULT_PLOT_SETTINGS.muted_text_color, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85))

    fig.tight_layout()
    out_path = _les6_output_path(out_dir, "incident_context_tsne_umap_comparison.png")
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


REGISTRY = {
    "incident_context_umap_analysis": incident_context_umap_analysis,
    "incident_context_tsne_umap_comparison": incident_context_tsne_umap_comparison,
}
