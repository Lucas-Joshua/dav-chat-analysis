"""Low-level plot constructors for chat activity visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS


def plot_chat_activity_by_hour(
    df: pd.DataFrame,
    out_path: str | Path = "img/chat_activity_by_hour.png",
):
    """Plot chat activity counts by hour of day."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")

    if "hour" not in df.columns:
        raise KeyError("hour column not found. Run add_time_features first.")

    # Count messages per hour
    hourly_counts = (
        df["hour"]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
        .reset_index()
    )

    hourly_counts.columns = ["hour", "count"]

    fig = px.line(
        hourly_counts,
        x="hour",
        y="count",
        markers=True,
        title="Distribution of Chat Messages Across Hours of the Day",
    )

    fig.update_traces(
        mode="lines",
        line=dict(width=3, color=DEFAULT_PLOT_SETTINGS.primary_color),
        marker=dict(color=DEFAULT_PLOT_SETTINGS.primary_color),
    )

    fig.update_xaxes(
        range=[0, 23],
        tickmode="linear",
        dtick=2,
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        constrain="domain"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        title="Number of Messages"
    )

    fig.update_layout(DEFAULT_PLOT_SETTINGS.base_plotly_layout())

    fig.add_annotation(
        x=12,
        y=hourly_counts["count"].max(),
        text="Peak activity",
        showarrow=True,
        arrowhead=2
    )

    fig.update_layout(
        title={
            "text": (
                "Distribution of Chat Messages Across Hours of the Day"
                f"<br><sup>Data from {date_min} – {date_max} · Aggregated Across All Days</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        }
    )

    fig.write_image(out_path, scale=2)


def plot_chat_activity_distribution(
    df: pd.DataFrame,
    output: Optional[Path] = None
) -> go.Figure:
    """
    Visualizes the distribution of chat messages across hours of the day.

    Insight:
    Most chat activity happens during operational hours of the dropzone.
    """

    df = df.copy()

    hourly_counts = (
        df.groupby("hour")
        .size()
        .reset_index(name="messages")
        .sort_values("hour")
    )

    fig = px.bar(
        hourly_counts,
        x="hour",
        y="messages",
        color_discrete_sequence=[DEFAULT_PLOT_SETTINGS.neutral_color],
        labels={
            "hour": "Hour of Day",
            "messages": "Number of Messages"
        },
    )

    fig.update_layout(
        template=DEFAULT_PLOT_SETTINGS.plotly_template,
        bargap=0.35,
        showlegend=False,
        title={
            "text": (
                "Most Chat Activity Happens During Operational Hours of the Dropzone"
                "<br><sup>Distribution of chat messages across hours of the day "
                "(Oct 2024 – Feb 2026)</sup>"
            ),
            "x": 0.5
        }
    )

    fig.update_xaxes(dtick=3)
    fig.update_yaxes(showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, zeroline=False)

    if output:
        fig.write_image(output)

    return fig
