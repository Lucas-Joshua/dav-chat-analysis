"""Low-level plot constructors for emoji-focused visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import plotly.express as px

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import (
    ensure_parent_dir,
    focus_colors,
    get_user_col,
    set_plotly_title,
    style_plotly_xy_axes,
)

_EMOJI_GROUP_LABELS = {
    "humor": "Humor",
    "positive": "Positive",
    "negative_reflective": "Negative reflective",
    "social": "Social",
}


def plot_overall_emoji_distribution(
        df: pd.DataFrame,
        out_path: str | Path = "img/overall_emoji_distribution.png",
):
    """Plot overall emoji-group distribution across the full dataset.

    :param df: Input dataframe containing ``emoji_group``.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = ensure_parent_dir(out_path)

    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    df = df[df["emoji_group"].notna()].copy()

    counts = (
        df.groupby("emoji_group")
        .size()
        .reset_index(name="count")
    )

    total = counts["count"].sum()
    counts["proportion"] = counts["count"] / total

    counts = counts.sort_values("proportion", ascending=False)
    counts["emoji_group_label"] = counts["emoji_group"].map(_EMOJI_GROUP_LABELS).fillna(counts["emoji_group"])

    # Per-category colors — match the brand identity from plot_settings
    _CATEGORY_COLORS = {
        "Humor":              DEFAULT_PLOT_SETTINGS.emoji_group_colors["humor"],           # geel
        "Positive":           DEFAULT_PLOT_SETTINGS.emoji_group_colors["positive"],        # groen
        "Negative reflective": DEFAULT_PLOT_SETTINGS.emoji_group_colors["negative_reflective"],  # rood
        "Social":             DEFAULT_PLOT_SETTINGS.emoji_group_colors["social"],          # blauw
    }

    top_prop = float(counts["proportion"].max())
    humor_prop = float(counts.loc[counts["emoji_group_label"] == "Humor", "proportion"].iloc[0]) \
        if "Humor" in counts["emoji_group_label"].values else top_prop
    positive_prop = float(counts.loc[counts["emoji_group_label"] == "Positive", "proportion"].iloc[0]) \
        if "Positive" in counts["emoji_group_label"].values else 0.0
    humor_plus_positive = humor_prop + positive_prop

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    bar_colors = [
        _CATEGORY_COLORS.get(label, DEFAULT_PLOT_SETTINGS.neutral_color)
        for label in counts["emoji_group_label"]
    ]
    bars = ax.bar(
        counts["emoji_group_label"],
        counts["proportion"],
        color=bar_colors,
        alpha=0.88,
        width=0.55,
    )
    for bar, proportion in zip(bars, counts["proportion"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            proportion + 0.012,
            f"{proportion:.1%}",
            ha="center",
            va="bottom",
            fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize + 1,
            fontweight="semibold",
            color=DEFAULT_PLOT_SETTINGS.text_color,
        )

    # Guiding annotation: show that humor + positive = social / light tone
    if humor_plus_positive > 0:
        ax.annotate(
            f"Humor + Positive samen: {humor_plus_positive:.0%}\n"
            "→ de groep communiceert luchtig, niet informatief",
            xy=("Positive", positive_prop),
            xytext=(1.55, top_prop * 0.72),
            fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize,
            color=DEFAULT_PLOT_SETTINGS.muted_text_color,
            arrowprops=dict(
                arrowstyle="->,head_width=0.25",
                color=DEFAULT_PLOT_SETTINGS.muted_text_color,
                lw=0.9,
                connectionstyle="arc3,rad=-0.25",
            ),
            bbox=DEFAULT_PLOT_SETTINGS.annotation_box,
        )

    ax.set_title("Humor domineert de chat — 7 op 10 emoji zijn luchtig of positief")
    ax.set_ylabel("Aandeel van alle emoji")
    ax.set_xlabel("Emoji-categorie")
    ax.set_ylim(0, max(0.60, top_prop + 0.12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    n_total = int(df.shape[0])
    ax.text(
        0.98, 0.98,
        f"n={n_total:,} emoji-berichten".replace(",", "."),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def plot_emoji_heatmap_png(
    df: pd.DataFrame,
    out_path: str | Path = "img/emoji_heatmap.png",
    top_n_emojis: int = 6,
    top_n_users: int = 6,
    user_col: Optional[str] = None,
):
    """Plot a user-by-emoji heatmap for the most-used emojis.

    :param df: Input dataframe containing ``emoji_list``.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :param top_n_emojis: Number of top emojis to include.
    :type top_n_emojis: int
    :param top_n_users: Number of most-active users to keep in the heatmap.
    :type top_n_users: int
    :param user_col: Optional user column override.
    :type user_col: Optional[str]
    :return: None.
    :rtype: None
    """
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
    top_users = (
        filtered[user_col]
        .value_counts()
        .head(top_n_users)
        .index
        .tolist()
    )
    filtered = filtered[filtered[user_col].isin(top_users)]

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
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Blues",
        title=(
            f"Emoji-gebruik verschilt per gebruiker"
            f"<br><sup>Top {top_n_users} actieve gebruikers × top {top_n_emojis} emoji</sup>"
        ),
    )

    fig.update_layout(
        **DEFAULT_PLOT_SETTINGS.base_plotly_layout(),
    )
    style_plotly_xy_axes(
        fig,
        x_title="Emoji",
        y_title="Gebruiker",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.write_image(out_path, scale=2)


def plot_emoji_usage_by_hour(
    df: pd.DataFrame,
    output: Optional[Path] = None
):
    """Visualize probability of emoji usage across hours of the day.

    :param df: Input dataframe containing ``hour`` and ``has_emoji``.
    :type df: pd.DataFrame
    :param output: Optional output image path.
    :type output: Optional[Path]
    :return: Plotly figure with hourly emoji probability.
    :rtype: plotly.graph_objs._figure.Figure
    """

    df = df.copy()

    hourly_emoji = (
        df.groupby("hour")["has_emoji"]
        .mean()
        .reset_index(name="emoji_probability")
        .sort_values("hour")
    )

    peak_idx = int(hourly_emoji["emoji_probability"].idxmax())
    peak_hour = int(hourly_emoji.loc[peak_idx, "hour"])
    peak_prob = float(hourly_emoji.loc[peak_idx, "emoji_probability"])

    fig = px.bar(
        hourly_emoji,
        x="hour",
        y="emoji_probability",
        color_discrete_sequence=[DEFAULT_PLOT_SETTINGS.neutral_color],
        labels={
            "hour": "Uur van de dag",
            "emoji_probability": "Kans op emoji in bericht"
        },
    )

    fig.add_scatter(
        x=hourly_emoji["hour"],
        y=hourly_emoji["emoji_probability"].rolling(window=3, center=True, min_periods=1).mean(),
        mode="lines",
        line=dict(color=DEFAULT_PLOT_SETTINGS.primary_color, width=2.3),
        name="3-uurs trend",
    )
    fig.add_vline(
        x=peak_hour,
        line_dash="dash",
        line_color=DEFAULT_PLOT_SETTINGS.danger_color,
        line_width=1.3,
    )
    fig.add_annotation(
        x=peak_hour,
        y=peak_prob,
        text=f"Piekuur: {peak_hour}:00 ({peak_prob:.1%})",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-35,
        font=dict(size=10, color=DEFAULT_PLOT_SETTINGS.danger_color),
    )

    fig.update_layout(
        template=DEFAULT_PLOT_SETTINGS.plotly_template,
        bargap=0.35,
        showlegend=False,
        title={
            "text": (
                "Emoji-gebruik door de dag heen · kans per uur"
                "<br><sup>Kans dat een bericht emoji bevat per uur "
                "· staaf = observatie, lijn = 3-uurs trend, stippellijn = piek</sup>"
            ),
            "x": 0.5
        }
    )

    fig.update_xaxes(dtick=3)
    fig.update_yaxes(range=[0, 1], showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, zeroline=False)

    if output:
        fig.write_image(output)

    return fig
