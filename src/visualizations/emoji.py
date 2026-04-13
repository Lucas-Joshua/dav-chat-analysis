"""Public emoji visualization wrappers and registry entries."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._emoji_plotters import (
    plot_emoji_heatmap_png,
    plot_emoji_usage_by_hour,
    plot_overall_emoji_distribution,
)
from src.visualizations.utils import resolve_lesson_output_path


def overall_emoji_distribution(df, out_dir: str | Path | None = None) -> None:
    """Generate the overall emoji distribution bar chart.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_overall_emoji_distribution(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "overall_emoji_distribution",
            "overall_emoji_distribution.png",
        ),
    )


def emoji_heatmap(df, out_dir: str | Path | None = None) -> None:
    """Generate the emoji usage heatmap for top emojis per user.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_emoji_heatmap_png(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "emoji_heatmap",
            "top_emojis_per_user.png",
        ),
    )


def emoji_usage_by_hour(df, out_dir: str | Path | None = None) -> None:
    """Generate the probability of emoji usage by hour.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
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
    "emoji_heatmap": emoji_heatmap,
    "emoji_usage_by_hour": emoji_usage_by_hour,
}
