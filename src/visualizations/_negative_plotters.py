"""Low-level plot constructors for negative-reaction emoji analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualizations.utils import ensure_parent_dir, get_user_col, top_user_order


def _emoji_group_counts(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    """Build per-user per-group emoji counts."""
    working = df[df["emoji_group"].notna()].copy()
    return (
        working.groupby([user_col, "emoji_group"])
        .size()
        .reset_index(name="count")
    )


def _negative_stats(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    """Compute total and negative emoji counts plus ratio per user."""
    exploded = df[df["emoji_group"].notna()].explode("emoji_list").dropna(subset=["emoji_list"])
    total_emoji = exploded.groupby(user_col).size().rename("total_emoji")
    negative_emoji = (
        exploded[exploded["emoji_group"] == "negative_reflective"]
        .groupby(user_col)
        .size()
        .rename("negative_emoji")
    )
    stats = pd.concat([total_emoji, negative_emoji], axis=1).fillna(0)
    stats["ratio"] = stats["negative_emoji"] / stats["total_emoji"].replace(0, pd.NA)
    return stats.fillna(0)


def plot_negative_reaction_concentration(
    df: pd.DataFrame,
    out_path: str | Path = "img/negative_reaction_concentration.png",
    top_users: int = 10,
) -> None:
    """
    Visualize the proportion of negative-reaction emojis per user.

    Users are sorted descending by negative-reaction usage.
    """

    out_path = ensure_parent_dir(out_path)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    user_col = get_user_col(df)
    counts = _emoji_group_counts(df, user_col)
    selected_users = top_user_order(counts, user_col, top_users)
    counts = counts[counts[user_col].isin(selected_users)]

    # Convert to proportions
    group_totals = counts.groupby(user_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / group_totals

    # Keep only negative_reaction group
    negative = counts[counts["emoji_group"] == "negative_reflective"].copy()

    # Sort descending
    negative = negative.sort_values("proportion", ascending=False)

    # Anonymize users
    negative[user_col] = [
        f"User {chr(65+i)}" for i in range(len(negative))
    ]

    negative[user_col] = pd.Categorical(
        negative[user_col],
        categories=negative[user_col].tolist(),
        ordered=True
    )

    max_prop = negative["proportion"].max()
    max_percent = round(max_prop * 100, 1)

    # Plot
    fig = px.bar(
        negative,
        y=user_col,
        x="proportion",
        orientation="h",
        color_discrete_sequence=["#DB4437"],
    )

    fig.update_layout(
        title=dict(
            text=(
                "Negative-Reaction Emoji Usage Is Concentrated Among a Few Users"
                f"<br><span style='font-size:14px;'>"
                f"Highest user: {max_percent}% of their emoji usage"
                "</span>"
            ),
            x=0,
            xanchor="left",
            pad=dict(l=120)
        ),
        xaxis_title=dict(text="Proportion of Emoji Usage"),
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=120, r=40, t=130, b=60),
        height=700,
        font=dict(size=14),
        showlegend=False
    )

    fig.update_xaxes(
        tickformat=".0%",
        range=[0, max_prop * 1.15],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        automargin=True,
        ticklabelstandoff=20,
        showgrid=False
    )

    fig.write_image(out_path, scale=2)


def plot_negative_reaction_diagnostic(
    df: pd.DataFrame,
    out_path: str | Path = "img/negative_reaction_diagnostic.png",
    top_users: int = 10,
) -> None:
    """
    Plot diagnostic metrics for negative reactions by user.

    - Red bars: proportion of negative-reaction emojis
    - Black dots: total emoji usage per user
    """

    out_path = ensure_parent_dir(out_path)
    user_col = get_user_col(df)
    stats = _negative_stats(df, user_col)

    # Filter top users by total emoji usage
    stats = stats.sort_values("total_emoji", ascending=False).head(top_users)

    # Sort by ratio descending
    stats = stats.sort_values("ratio", ascending=False)

    # Anonymize
    stats["anon"] = [f"User {chr(65+i)}" for i in range(len(stats))]

    # Build figure
    fig = go.Figure()

    # Red bars (ratio)
    fig.add_trace(go.Bar(
        y=stats["anon"],
        x=stats["ratio"],
        orientation="h",
        name="Negative Reaction Ratio",
        marker=dict(color="#DB4437"),
    ))

    # Black dots (total emoji)
    fig.add_trace(go.Scatter(
        y=stats["anon"],
        x=stats["total_emoji"],
        mode="markers",
        name="Total Emoji Used",
        marker=dict(color="black", size=8),
        xaxis="x2"
    ))

    fig.update_layout(
        title=dict(
            text="Negative-Reaction Proportion vs Total Emoji Usage",
            x=0,
            xanchor="left"
        ),
        xaxis=dict(
            title=dict(text="Proportion of Emoji Usage"),
            tickformat=".0%",
            range=[0, stats["ratio"].max() * 1.2]
        ),
        xaxis2=dict(
            title=dict(text="Total Emoji Count"),
            overlaying="x",
            side="top"
        ),
        yaxis=dict(
            autorange="reversed"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=700,
        margin=dict(l=120, r=40, t=120, b=60)
    )

    fig.write_image(out_path, scale=2)


def plot_negative_reaction_scatter(
    df: pd.DataFrame,
    out_path: str | Path = "img/negative_reaction_scatter.png",
) -> None:
    """Scatter total emoji usage versus negative reaction ratio."""
    out_path = ensure_parent_dir(out_path)
    user_col = get_user_col(df)
    stats = _negative_stats(df, user_col)
    stats_reset = stats.reset_index().rename(columns={user_col: "user"})

    fig = px.scatter(
        stats_reset,
        x="total_emoji",
        y="ratio",
        hover_name="user",
    )

    fig.update_layout(
        title=dict(text="Negative-Reaction Usage Decreases as Emoji Volume Increases"),
        xaxis_title=dict(text="Total Emoji Used"),
        yaxis_title=dict(text="Proportion of Negative-Reaction Emoji"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
    )

    fig.update_yaxes(tickformat=".0%")

    fig.write_image(out_path, scale=2)
