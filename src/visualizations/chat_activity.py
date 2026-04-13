"""Public chat-activity visualization wrappers and registry entries."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._activity_plotters import (
    plot_chat_activity_by_hour,
    plot_chat_activity_weekday_weekend,
)
from src.visualizations.utils import resolve_lesson_output_path


def chat_activity_by_hour(df, out_dir: str | Path | None = None) -> None:
    """Generate the chat activity by-hour line chart.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_chat_activity_by_hour(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "chat_activity_by_hour",
            "chat_activity_by_hour.png",
        ),
    )


def chat_activity_weekday_weekend(df, out_dir: str | Path | None = None) -> None:
    """Generate weekday-vs-weekend daily activity comparison.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_chat_activity_weekday_weekend(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "chat_activity_weekday_weekend",
            "chat_activity_weekday_weekend.png",
        ),
    )


REGISTRY = {
    "chat_activity_by_hour": chat_activity_by_hour,
    "chat_activity_weekday_weekend": chat_activity_weekday_weekend,
}
