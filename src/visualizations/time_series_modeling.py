"""Public wrappers and registry for time-series modeling visualizations (Les 4)."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._time_series_plotters import (
    plot_autocorrelation,
    plot_time_series_activity,
)
from src.visualizations.utils import resolve_lesson_output_path


def time_series_activity(df, out_dir: str | Path | None = None) -> None:
    """Generate the 15-minute time series activity chart with trend and residuals.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_time_series_activity(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "time_series_activity",
            "time_series_activity.png",
        ),
    )


def time_series_autocorrelation(df, out_dir: str | Path | None = None) -> None:
    """Generate the autocorrelation chart for 15-minute message counts.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_autocorrelation(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "time_series_autocorrelation",
            "time_series_autocorrelation.png",
        ),
    )


REGISTRY = {
    "time_series_activity": time_series_activity,
    "time_series_autocorrelation": time_series_autocorrelation,
}
