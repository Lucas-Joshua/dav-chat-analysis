"""Emoji visualizations and registry entries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import plotly.express as px

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import (
    ensure_parent_dir,
    resolve_lesson_output_path,
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
) -> None:
    """Plot overall emoji-group distribution across the full dataset."""
    out_path = ensure_parent_dir(out_path)
    if "emoji_group" not in df.columns:
        raise KeyError("emoji_group column not found.")

    df = df[df["emoji_group"].notna()].copy()
    counts = df.groupby("emoji_group").size().reset_index(name="count")
    total = counts["count"].sum()
    counts["proportion"] = counts["count"] / total
    counts = counts.sort_values("proportion", ascending=False)
    counts["emoji_group_label"] = counts["emoji_group"].map(_EMOJI_GROUP_LABELS).fillna(counts["emoji_group"])

    category_colors = {
        "Humor": DEFAULT_PLOT_SETTINGS.emoji_group_colors["humor"],
        "Positive": DEFAULT_PLOT_SETTINGS.emoji_group_colors["positive"],
        "Negative reflective": DEFAULT_PLOT_SETTINGS.emoji_group_colors["negative_reflective"],
        "Social": DEFAULT_PLOT_SETTINGS.emoji_group_colors["social"],
    }

    top_prop = float(counts["proportion"].max())
    humor_prop = float(counts.loc[counts["emoji_group_label"] == "Humor", "proportion"].iloc[0]) if "Humor" in counts["emoji_group_label"].values else top_prop
    positive_prop = float(counts.loc[counts["emoji_group_label"] == "Positive", "proportion"].iloc[0]) if "Positive" in counts["emoji_group_label"].values else 0.0
    humor_plus_positive = humor_prop + positive_prop

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    bar_colors = [category_colors.get(label, DEFAULT_PLOT_SETTINGS.neutral_color) for label in counts["emoji_group_label"]]
    bars = ax.bar(counts["emoji_group_label"], counts["proportion"], color=bar_colors, alpha=0.88, width=0.55)
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

    if humor_plus_positive > 0:
        ax.annotate(
            f"Humor + Positive samen: {humor_plus_positive:.0%}\n→ de groep communiceert luchtig, niet informatief",
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
        0.98,
        0.98,
        f"n={n_total:,} emoji-berichten".replace(",", "."),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def plot_emoji_usage_by_hour(
    df: pd.DataFrame,
    output: Path | None = None,
):
    """Visualize probability of emoji usage across hours of the day."""
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
        labels={"hour": "Uur van de dag", "emoji_probability": "Kans op emoji in bericht"},
    )
    fig.add_scatter(
        x=hourly_emoji["hour"],
        y=hourly_emoji["emoji_probability"].rolling(window=3, center=True, min_periods=1).mean(),
        mode="lines",
        line=dict(color=DEFAULT_PLOT_SETTINGS.primary_color, width=2.3),
        name="3-uurs trend",
    )
    fig.add_vline(x=peak_hour, line_dash="dash", line_color=DEFAULT_PLOT_SETTINGS.danger_color, line_width=1.3)
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
            "x": 0.5,
        },
    )
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(range=[0, 1], showgrid=True, gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor, zeroline=False)

    if output:
        fig.write_image(output)
    return fig


def overall_emoji_distribution(df, out_dir: str | Path | None = None) -> None:
    """Generate the overall emoji distribution bar chart."""
    plot_overall_emoji_distribution(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "overall_emoji_distribution",
            "overall_emoji_distribution.png",
        ),
    )
def emoji_usage_by_hour(df, out_dir: str | Path | None = None) -> None:
    """Generate the probability of emoji usage by hour."""
    plot_emoji_usage_by_hour(
        df,
        output=resolve_lesson_output_path(
            out_dir,
            "emoji_usage_by_hour",
            "plot_emoji_usage_by_hour.png",
        ),
    )


REGISTRY = {
    "overall_emoji_distribution": overall_emoji_distribution,
    "emoji_usage_by_hour": emoji_usage_by_hour,
}
