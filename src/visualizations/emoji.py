"""Public emoji visualization wrappers and registry entries."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._emoji_plotters import (
    plot_emoji_heatmap_png,
    plot_emoji_type_per_user,
    plot_emoji_usage_by_hour,
    plot_overall_emoji_distribution,
)
from src.visualizations.utils import resolve_output_path


def overall_emoji_distribution(df, out_dir: str | Path | None = None) -> None:
    """Generate the overall emoji distribution bar chart."""
    plot_overall_emoji_distribution(
        df,
        out_path=resolve_output_path(out_dir, "overall_emoji_distribution.png"),
    )


def emoji_heatmap(df, out_dir: str | Path | None = None) -> None:
    """Generate the emoji usage heatmap for top emojis per user."""
    plot_emoji_heatmap_png(
        df,
        out_path=resolve_output_path(out_dir, "emoji_heatmap.png"),
    )


def emoji_type_per_user(df, out_dir: str | Path | None = None) -> None:
    """Generate the emoji group distribution per user."""
    plot_emoji_type_per_user(
        df,
        out_path=resolve_output_path(out_dir, "emoji_group_distribution.png"),
    )


def emoji_usage_by_hour(df, out_dir: str | Path | None = None) -> None:
    """Generate the probability of emoji usage by hour."""
    plot_emoji_usage_by_hour(
        df,
        output=Path(resolve_output_path(out_dir, "plot_emoji_usage_by_hour.png")),
    )


REGISTRY = {
    "overall_emoji_distribution": overall_emoji_distribution,
    "emoji_heatmap": emoji_heatmap,
    "emoji_type_per_user": emoji_type_per_user,
    "emoji_usage_by_hour": emoji_usage_by_hour,
}
