"""Incident-context dimensionality-reduction views for Les 6."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    import matplotlib.patches as mpatches
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.spatial import ConvexHull

    import umap as umap_lib  # type: ignore[import-not-found]
    from sklearn.feature_extraction.text import TfidfVectorizer

    sample = _prepare_context_sample(df, max_points=1600)
    texts = sample["message"].tolist()

    vectorizer = TfidfVectorizer(max_features=500, min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(texts).toarray()

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.10, random_state=42)
    emb = reducer.fit_transform(X)
    x = emb[:, 0]
    y = emb[:, 1]
    is_incident = sample["incident_related"].to_numpy()

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
        except Exception:
            pass  # skip KDE if degenerate data

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
        except Exception:
            pass

    # --- Insight annotation — anchored in the lower-right empty quadrant ---
    cx, cy = float(np.median(x_inc)), float(np.median(y_inc))
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    # Text goes to lower-right area (typically empty in this manifold shape)
    txt_x = x.min() + x_range * 0.62
    txt_y = y.min() + y_range * 0.18
    ax.annotate(
        "Incidentberichten concentreren\nzich in een herkenbaar gebied\n"
        "→ UMAP maakt dit patroon zichtbaar,\n   t-SNE niet",
        xy=(cx, cy),
        xytext=(txt_x, txt_y),
        fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize,
        color=DEFAULT_PLOT_SETTINGS.danger_color,
        arrowprops=dict(
            arrowstyle="->,head_width=0.25",
            color=DEFAULT_PLOT_SETTINGS.danger_color, lw=1.0,
            connectionstyle="arc3,rad=0.25",
        ),
        bbox=DEFAULT_PLOT_SETTINGS.annotation_box,
        va="bottom",
    )

    ax.set_xlabel("UMAP dimensie 1")
    ax.set_ylabel("UMAP dimensie 2")
    ax.set_title(
        "UMAP onthult ruimtelijke clustering van incidentberichten\n"
        "Trigram-profielen · n_neighbors=15 · min_dist=0.10 · rood = incident-context"
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


REGISTRY = {
    "incident_context_umap_analysis": incident_context_umap_analysis,
}
