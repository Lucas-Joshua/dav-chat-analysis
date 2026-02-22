from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px


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


def plot_top_emojis_per_user_png(
    df_top_user: pd.DataFrame,
    out_path: str | Path = "img/top_emojis_per_user.png",
    top_users: int = 10,
):

    out_path = _ensure_parent_dir(out_path)
    user_col = _get_user_col(df_top_user)

    top_user_order = (
        df_top_user.groupby(user_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_users)
        .index
        .tolist()
    )

    df_plot = df_top_user[df_top_user[user_col].isin(top_user_order)].copy()

    df_plot[user_col] = pd.Categorical(
        df_plot[user_col],
        categories=top_user_order,
        ordered=True
    )

    fig = px.bar(
        df_plot,
        x=user_col,
        y="count",
        color="emoji_list",
        barmode="stack",
        title=f"Top Emoji Usage per Top {top_users} Users"
    )

    fig.update_layout(
        xaxis_title="User",
        yaxis_title="Emoji count",
        legend_title="Emoji",
        font=dict(size=14),
    )

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