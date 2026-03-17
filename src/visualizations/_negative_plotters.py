from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_negative_reaction_concentration(
    df: pd.DataFrame,
    out_path: str | Path = "img/negative_reaction_concentration.png",
    top_users: int = 10,
):
    """
    Visualizes the proportion of negative-reaction emojis per user.
    Users are sorted descending by negative-reaction usage.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    user_col = "user" if "user" in df.columns else "sender"

    df = df[df["emoji_group"].notna()].copy()

    # Count per user per group
    counts = (
        df.groupby([user_col, "emoji_group"])
        .size()
        .reset_index(name="count")
    )

    # Top users by total emoji usage
    top_user_order = (
        counts.groupby(user_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_users)
        .index
        .tolist()
    )

    counts = counts[counts[user_col].isin(top_user_order)]

    # Convert to proportions
    counts["proportion"] = (
        counts.groupby(user_col)["count"]
        .transform(lambda x: x / x.sum())
    )

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
        xaxis_title="Proportion of Emoji Usage",
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
):
    """
    Diagnostic plot:
    - Red bars: proportion of negative-reaction emojis
    - Black dots: total emoji usage per user
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    user_col = "user" if "user" in df.columns else "sender"

    df = df[df["emoji_group"].notna()].copy()

    # Total emoji per user
    total_emoji = (
        df.groupby(user_col)["emoji_list"]
        .apply(lambda x: sum(len(i) for i in x))
    )

    # Negative emoji per user
    negative_emoji = (
        df[df["emoji_group"] == "negative_reflective"]
        .groupby(user_col)
        .size()
    )

    stats = (
        pd.DataFrame({
            "total_emoji": total_emoji,
            "negative_emoji": negative_emoji
        })
        .fillna(0)
    )

    stats["ratio"] = stats["negative_emoji"] / stats["total_emoji"]

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
        marker_color="#DB4437"
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
            title="Proportion of Emoji Usage",
            tickformat=".0%",
            range=[0, stats["ratio"].max() * 1.2]
        ),
        xaxis2=dict(
            title="Total Emoji Count",
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
):
    """Scatter total emoji usage versus negative reaction ratio."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    user_col = "user" if "user" in df.columns else "sender"

    df = df[df["emoji_group"].notna()].copy()

    total_emoji = (
        df.groupby(user_col)["emoji_list"]
        .apply(lambda x: sum(len(i) for i in x))
    )

    negative_emoji = (
        df[df["emoji_group"] == "negative_reflective"]
        .groupby(user_col)
        .size()
    )

    stats = (
        pd.DataFrame({
            "total_emoji": total_emoji,
            "negative_emoji": negative_emoji
        })
        .fillna(0)
    )

    stats["ratio"] = stats["negative_emoji"] / stats["total_emoji"]

    fig = px.scatter(
        stats,
        x="total_emoji",
        y="ratio",
        hover_name=stats.index,
    )

    fig.update_layout(
        title="Negative-Reaction Usage Decreases as Emoji Volume Increases",
        xaxis_title="Total Emoji Used",
        yaxis_title="Proportion of Negative-Reaction Emoji",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
    )

    fig.update_yaxes(tickformat=".0%")

    fig.write_image(out_path, scale=2)
