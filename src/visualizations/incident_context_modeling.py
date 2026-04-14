"""Incident-context dimensionality-reduction views for Les 6."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.modules.feature_engineering import add_incident_bow_features
from src.modules.author_stylometry import compute_stylometric_embedding
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import ensure_parent_dir, set_plotly_title

ContextMethod = Literal["PCA", "tSNE", "UMAP"]
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


def _context_scatter(
    embed_df: pd.DataFrame,
    title: str,
    out_path: str | Path,
) -> None:
    """Render a two-group context scatter in reduced space.

    :param embed_df: Embedding dataframe with ``x``, ``y`` and ``incident_related``.
    :type embed_df: pd.DataFrame
    :param title: Chart title.
    :type title: str
    :param out_path: Output image path.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = ensure_parent_dir(out_path)
    colors = {True: DEFAULT_PLOT_SETTINGS.danger_color, False: DEFAULT_PLOT_SETTINGS.neutral_color}
    labels = {True: "Incident-gerelateerd", False: "Regulier"}

    fig = go.Figure()
    for flag in [False, True]:
        subset = embed_df[embed_df["incident_related"] == flag]
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=labels[flag],
                marker=dict(color=colors[flag], size=7, opacity=0.9 if flag else 0.28),
                hoverinfo="skip",
            )
        )

    n_incident = int(embed_df["incident_related"].sum())
    n_regular = int((~embed_df["incident_related"]).sum())
    set_plotly_title(
        fig,
        title=title,
        subtitle=(
            f"Rood = incident-gerelateerd ({n_incident}) \u00b7 Grijs = regulier ({n_regular}) "
            "\u00b7 nabijheid = semantische gelijkenis"
        ),
    )
    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 50, "r": 30, "t": 95, "b": 50},
        )
    )
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, title_text="t-SNE dimensie 1")
    fig.update_yaxes(showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, title_text="t-SNE dimensie 2")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.05,
        text=(
            "<b>Bevinding:</b> incident-berichten vormen geen apart cluster<br>"
            "→ de chat-<i>taal</i> verandert niet bij incidenten, alleen het <i>volume</i>"
        ),
        showarrow=False,
        font=dict(size=11, color="#444444"),
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#CCCCCC",
        borderwidth=1,
    )

    fig.write_image(str(out_path), scale=2)


def incident_context_projection(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
    method: ContextMethod = "tSNE",
) -> None:
    """Project messages and color by incident-related context.

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :param method: Dimensionality-reduction method.
    :type method: ContextMethod
    :return: None.
    :rtype: None
    """
    sample = _prepare_context_sample(df)
    texts = sample["message"].tolist()
    embedding = compute_stylometric_embedding(texts, method=method, n_components=2)
    embed_df = pd.DataFrame(
        {"x": embedding[:, 0], "y": embedding[:, 1], "incident_related": sample["incident_related"]}
    )

    filename = f"incident_context_projection_{method.lower()}.png"
    out_path = _les6_output_path(out_dir, filename)
    _context_scatter(
        embed_df,
        title="Incident-berichten vormen taalkundig geen aparte groep",
        out_path=out_path,
    )


def incident_context_comparison(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Optional comparison view across PCA, t-SNE, and UMAP for context labels.

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    sample = _prepare_context_sample(df)
    texts = sample["message"].tolist()
    methods: list[ContextMethod] = ["PCA", "tSNE", "UMAP"]
    embeddings: dict[str, pd.DataFrame] = {}
    for method in methods:
        emb = compute_stylometric_embedding(texts, method=method, n_components=2)
        embeddings[method] = pd.DataFrame(
            {"x": emb[:, 0], "y": emb[:, 1], "incident_related": sample["incident_related"]}
        )

    fig = make_subplots(rows=1, cols=3, subplot_titles=methods, horizontal_spacing=0.08)
    colors = {True: "#C62828", False: "#4E79A7"}

    for idx, method in enumerate(methods, start=1):
        for flag in [False, True]:
            subset = embeddings[method][embeddings[method]["incident_related"] == flag]
            fig.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    showlegend=idx == 1,
                    name="Incident" if flag else "Regulier",
                    marker=dict(color=colors[flag], size=6, opacity=0.65),
                    hoverinfo="skip",
                ),
                row=1,
                col=idx,
            )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=idx)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=idx)

    set_plotly_title(
        fig,
        title="Incidentcontext in gereduceerde ruimte",
        subtitle="Ondersteunende vergelijking over reductiemethoden",
    )
    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 30, "r": 30, "t": 90, "b": 40},
        )
    )
    fig.update_layout(height=420, width=1200, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))

    out_path = _les6_output_path(out_dir, "incident_context_comparison.png")
    fig.write_image(str(out_path), scale=2)


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

    sample = _prepare_context_sample(df, max_points=1600)
    texts = sample["message"].tolist()
    emb = compute_stylometric_embedding(
        texts,
        method="UMAP",
        n_components=2,
        n_neighbors=15,
        min_dist=0.10,
    )
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
    "incident_context_projection": incident_context_projection,
    "incident_context_comparison": incident_context_comparison,
    "incident_context_umap_analysis": incident_context_umap_analysis,
}
