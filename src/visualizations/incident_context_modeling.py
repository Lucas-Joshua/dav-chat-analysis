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
            f"Rood = incident-gerelateerd ({n_incident}) · Grijs = regulier ({n_regular})"
        ),
    )
    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 50, "r": 30, "t": 95, "b": 50},
        )
    )
    fig.update_layout(height=520, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
    fig.update_xaxes(showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, title_text="")
    fig.update_yaxes(showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, title_text="")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        text="Geen duidelijke scheiding zichtbaar",
        showarrow=False,
        font=dict(size=11, color="#5a5a5a"),
        align="left",
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
        title="Incident- en reguliere berichten in gereduceerde ruimte",
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


REGISTRY = {
    "incident_context_projection": incident_context_projection,
    "incident_context_comparison": incident_context_comparison,
}
