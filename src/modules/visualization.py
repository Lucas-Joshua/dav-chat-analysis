from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _ensure_parent_dir(out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def _get_user_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    if "user" in df.columns:
        return "user"
    if "sender" in df.columns:
        return "sender"
    raise KeyError("No user column found.")

def plot_negative_reaction_concentration(
    df: pd.DataFrame,
    out_path: str | Path = "img/negative_reaction_concentration.png",
    top_users: int = 10,
):
    """
    Visualizes the proportion of negative-reaction emojis per user.
    Users are sorted descending by negative-reaction usage.
    """

    from pathlib import Path
    import pandas as pd
    import plotly.express as px

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
    from pathlib import Path
    import pandas as pd
    import plotly.express as px

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

def plot_overall_emoji_distribution(
        df: pd.DataFrame,
        out_path: str | Path = "img/overall_emoji_distribution.png",
):
    """
    Plots overall emoji group distribution across the full dataset.
    """

    from pathlib import Path
    import pandas as pd
    import plotly.express as px

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    color_map = {
        "humor": "#F4B400",
        "positive": "#34A853",
        "social": "#4285F4",
        "negative_reflective": "#DB4437"
    }

    fig = px.bar(
        counts,
        x="emoji_group",
        y="proportion",
        color="emoji_group",
        color_discrete_map=color_map,
    )

    fig.update_layout(
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=120, b=60),
        height=500,
        showlegend=False,
    )

    fig.update_yaxes(
        tickformat=".0%",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
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

    out_path = _ensure_parent_dir(out_path)
    user_col = _get_user_col(df, preferred=user_col)

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
        font=dict(size=14),
    )

    fig.write_image(out_path, scale=2)


def plot_emoji_type_per_user(
    df: pd.DataFrame,
    out_path: str | Path = "img/emoji_group_distribution.png",
    top_users: int = 10,
):

    out_path = _ensure_parent_dir(out_path)
    user_col = _get_user_col(df)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    df = df[df["emoji_group"] != "other"].copy()

    counts = (
        df.groupby([user_col, "emoji_group"])
        .size()
        .reset_index(name="count")
    )

    top_user_order = (
        counts.groupby(user_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_users)
        .index
        .tolist()
    )

    counts = counts[counts[user_col].isin(top_user_order)]

    counts["count"] = (
        counts.groupby(user_col)["count"]
        .transform(lambda x: x / x.sum())
    )

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

    color_map = {
        "humor": "#F4B400",
        "positive": "#34A853",
        "social": "#4285F4",
        "negative_reflective": "#DB4437"
    }

    fig = px.bar(
        counts,
        y=user_col,
        x="count",
        color="emoji_group",
        orientation="h",
        barmode="stack",
        color_discrete_map=color_map,
        title="Comparing Communicative Styles Across Top 10 Users",
    )

    fig.update_layout(
        xaxis_title="Percentage of Emoji Usage",
        yaxis_title="User",
        legend_title="Communicative Style",
        font=dict(size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.write_image(out_path, scale=2)


def plot_chat_activity_by_hour(
    df: pd.DataFrame,
    out_path: str | Path = "img/chat_activity_by_hour.png",
):
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
        line=dict(width=3)
    )

    fig.update_xaxes(
        range=[0, 23],
        tickmode="linear",
        dtick=2,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        constrain="domain"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        title="Number of Messages"
    )

    fig.update_layout(
        font=dict(size=14),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

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
        color_discrete_sequence=["#CFCFCF"],
        labels={
            "hour": "Hour of Day",
            "messages": "Number of Messages"
        },
    )

    fig.update_layout(
        template="simple_white",
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

    # mark peak hour
    peak_row = hourly_counts.loc[hourly_counts["messages"].idxmax()]

    if output:
        fig.write_image(output)

    return fig


def plot_emoji_usage_by_hour(
    df: pd.DataFrame,
    output: Optional[Path] = None
) -> go.Figure:
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
        color_discrete_sequence=["#CFCFCF"],
        labels={
            "hour": "Hour of Day",
            "emoji_probability": "Probability of Emoji in Message"
        },
    )

    fig.update_layout(
        template="simple_white",
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
    fig.update_yaxes(range=[0, 1])

    if output:
        fig.write_image(output)

    return fig

