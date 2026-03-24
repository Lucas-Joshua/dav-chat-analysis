"""Low-level plot constructors for emoji-focused visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import ensure_parent_dir, get_user_col, top_user_order


def plot_overall_emoji_distribution(
        df: pd.DataFrame,
        out_path: str | Path = "img/overall_emoji_distribution.png",
):
    """Plot overall emoji group distribution across the full dataset."""
    out_path = ensure_parent_dir(out_path)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    df = df[df["emoji_group"].notna()].copy()

    # Count per group
    counts = (
        df.groupby("emoji_group")
        .size()
        .reset_index(name="count")
    )

    # Convert to proportions
    total = counts["count"].sum()
    counts["proportion"] = counts["count"] / total

    # Sort descending
    counts = counts.sort_values("proportion", ascending=False)

    fig = px.bar(
        counts,
        x="emoji_group",
        y="proportion",
        color="emoji_group",
        color_discrete_map=DEFAULT_PLOT_SETTINGS.emoji_group_colors,
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            title=dict(
                text=(
                    "Overall Emoji Usage Is Dominated by Humor"
                    "<br><span style='font-size:14px;'>"
                    "Distribution across entire dataset"
                    "</span>"
                ),
                x=0,
                xanchor="left",
                pad=dict(l=80)
            ),
            xaxis_title="Emoji Category",
            yaxis_title="Proportion of Emoji Usage",
            margin=dict(l=80, r=40, t=120, b=60),
            height=500,
            showlegend=False,
        )
    )

    fig.update_yaxes(
        tickformat=".0%",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False
    )

    fig.update_xaxes(showgrid=False)

    fig.write_image(out_path, scale=2)


def plot_emoji_heatmap_png(
    df: pd.DataFrame,
    out_path: str | Path = "img/emoji_heatmap.png",
    top_n_emojis: int = 10,
    user_col: Optional[str] = None,
):
    """Plot a user-by-emoji heatmap for the most-used emojis."""
    out_path = ensure_parent_dir(out_path)
    user_col = get_user_col(df, preferred=user_col)

    if "emoji_list" not in df.columns:
        raise KeyError("emoji_list column not found.")

    exploded = df.explode("emoji_list").dropna(subset=["emoji_list"])

    if exploded.empty:
        raise ValueError("No emojis available for heatmap.")

    top_emojis = (
        exploded["emoji_list"]
        .value_counts()
        .head(top_n_emojis)
        .index
        .tolist()
    )

    filtered = exploded[exploded["emoji_list"].isin(top_emojis)]

    heatmap_data = (
        filtered.groupby([user_col, "emoji_list"])
        .size()
        .unstack(fill_value=0)
    )

    heatmap_data = heatmap_data.loc[
        heatmap_data.sum(axis=1).sort_values(ascending=False).index,
        heatmap_data.sum(axis=0).sort_values(ascending=False).index,
    ]

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Emoji Usage Heatmap (Top {top_n_emojis})",
    )

    fig.update_layout(
        xaxis_title="Emoji",
        yaxis_title="User",
        **DEFAULT_PLOT_SETTINGS.base_plotly_layout(),
    )

    fig.write_image(out_path, scale=2)


def plot_emoji_type_per_user(
    df: pd.DataFrame,
    out_path: str | Path = "img/emoji_group_distribution.png",
    top_users: int = 10,
):
    """Plot per-user emoji group proportions for top contributors."""
    out_path = ensure_parent_dir(out_path)
    user_col = get_user_col(df)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    df = df[df["emoji_group"] != "other"].copy()

    counts = (
        df.groupby([user_col, "emoji_group"])
        .size()
        .reset_index(name="count")
    )

    selected_users = top_user_order(counts, user_col, top_users)

    counts = counts[counts[user_col].isin(selected_users)]

    group_totals = counts.groupby(user_col)["count"].transform("sum")
    counts["count"] = counts["count"] / group_totals

    dominant = (
        counts.sort_values("count", ascending=False)
        .groupby(user_col)
        .first()
    )

    order = dominant.sort_values("emoji_group").index.tolist()

    counts[user_col] = pd.Categorical(
        counts[user_col],
        categories=order,
        ordered=True
    )

    fig = px.bar(
        counts,
        y=user_col,
        x="count",
        color="emoji_group",
        orientation="h",
        barmode="stack",
        color_discrete_map=DEFAULT_PLOT_SETTINGS.emoji_group_colors,
        title="Comparing Communicative Styles Across Top 10 Users",
    )

    fig.update_layout(
        xaxis_title="Percentage of Emoji Usage",
        yaxis_title="User",
        legend_title="Communicative Style",
        **DEFAULT_PLOT_SETTINGS.base_plotly_layout(),
    )

    fig.write_image(out_path, scale=2)


def plot_emoji_usage_by_hour(
    df: pd.DataFrame,
    output: Optional[Path] = None
):
    """
    Visualizes the probability that a message contains emojis across hours of the day.

    Insight:
    Emoji usage increases during social hours after skydiving activities.
    """

    df = df.copy()

    hourly_emoji = (
        df.groupby("hour")["has_emoji"]
        .mean()
        .reset_index(name="emoji_probability")
        .sort_values("hour")
    )

    fig = px.bar(
        hourly_emoji,
        x="hour",
        y="emoji_probability",
        color_discrete_sequence=[DEFAULT_PLOT_SETTINGS.neutral_color],
        labels={
            "hour": "Hour of Day",
            "emoji_probability": "Probability of Emoji in Message"
        },
    )

    fig.update_layout(
        template=DEFAULT_PLOT_SETTINGS.plotly_template,
        bargap=0.35,
        showlegend=False,
        title={
            "text": (
                "Emoji Usage Increases During Social Hours After Jumping Activities"
                "<br><sup>Probability that a message contains emojis by hour of day "
                "(Oct 2024 – Feb 2026)</sup>"
            ),
            "x": 0.5
        }
    )

    fig.update_xaxes(dtick=3)
    fig.update_yaxes(range=[0, 1], showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, zeroline=False)

    if output:
        fig.write_image(output)

    return fig
