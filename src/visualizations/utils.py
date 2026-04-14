"""Shared utility helpers for visualization output management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS

VISUALIZATION_LESSON_MAP: dict[str, str] = {
    # Les 2: category/comparison
    "overall_emoji_distribution": "les2",
    "emoji_heatmap": "les2",
    # Les 3: time
    "chat_activity_by_hour": "les3",
    "emoji_usage_by_hour": "les3",
    # Les 4: distributions
    "chat_activity_weekday_weekend": "les4",
    "time_series_activity": "les4",
    "time_series_autocorrelation": "les4",
    "poisson_model": "les4",
    # Les 5: relationships
    "incident_discussion_timeline": "les5",
    "incident_activity_correlation": "les5",
    # Les 6: dimensionality reduction / modelling
    "incident_context_projection": "les6",
    "incident_context_comparison": "les6",
}


def resolve_output_path(out_dir: str | Path | None, filename: str) -> str | Path:
    """Build a plot output path using the configured output directory.

    :param out_dir: Optional output directory root.
    :type out_dir: str | Path | None
    :param filename: Plot filename.
    :type filename: str
    :return: Resolved output path.
    :rtype: str | Path
    """
    return Path(out_dir) / filename if out_dir else f"img/{filename}"


def resolve_lesson_output_path(
    out_dir: str | Path | None,
    visualization_name: str,
    filename: str,
) -> Path:
    """Resolve output path under the lesson folder for a visualization.

    :param out_dir: Optional output directory root.
    :type out_dir: str | Path | None
    :param visualization_name: Registry key of the visualization.
    :type visualization_name: str
    :param filename: Output filename.
    :type filename: str
    :return: Resolved lesson-scoped output path.
    :rtype: Path
    """
    root = Path(out_dir) if out_dir else Path("img")
    lesson = VISUALIZATION_LESSON_MAP.get(visualization_name)
    path = root / lesson / filename if lesson else root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(out_path: str | Path) -> Path:
    """Ensure the output directory exists and return the resolved path.

    :param out_path: Target file path.
    :type out_path: str | Path
    :return: Path object with ensured parent directory.
    :rtype: Path
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_user_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Resolve the user column name from the dataframe.

    :param df: Input dataframe with either ``user`` or ``sender`` column.
    :type df: pd.DataFrame
    :param preferred: Preferred column name when present.
    :type preferred: Optional[str]
    :return: Chosen user column name.
    :rtype: str
    """
    if preferred and preferred in df.columns:
        return preferred
    if "user" in df.columns:
        return "user"
    if "sender" in df.columns:
        return "sender"
    raise KeyError("No user column found.")


def top_user_order(counts: pd.DataFrame, user_col: str, top_users: int) -> list[str]:
    """Return top users ordered by descending total count.

    :param counts: Dataframe containing per-user counts.
    :type counts: pd.DataFrame
    :param user_col: Name of the user column in ``counts``.
    :type user_col: str
    :param top_users: Number of users to keep.
    :type top_users: int
    :return: Ordered list of top user identifiers.
    :rtype: list[str]
    """
    totals = counts.groupby(user_col, as_index=True)["count"].sum()
    return totals.sort_values(ascending=False).head(top_users).index.tolist()


def hourly_message_counts(df: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    """Return a complete 24-hour count table for message activity.

    :param df: Dataframe containing an hour column.
    :type df: pd.DataFrame
    :param hour_col: Name of the hour column.
    :type hour_col: str
    :return: Dataframe with ``hour`` and ``count`` columns.
    :rtype: pd.DataFrame
    """
    if hour_col not in df.columns:
        raise KeyError(f"{hour_col} column not found.")

    counts = (
        df[hour_col]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
        .reset_index()
    )
    counts.columns = ["hour", "count"]
    return counts


def set_plotly_title(
    fig,
    title: str,
    subtitle: str | None = None,
    x: float = 0.5,
) -> None:
    """Apply a consistent title/subtitle hierarchy to a Plotly figure.

    :param fig: Target Plotly figure.
    :type fig: Any
    :param title: Main title line.
    :type title: str
    :param subtitle: Optional subtitle line.
    :type subtitle: str | None
    :param x: Horizontal anchor position.
    :type x: float
    :return: None.
    :rtype: None
    """
    title_html = f"<b>{title}</b>"

    if subtitle is None:
        text = title_html
    else:
        subtitle_html = subtitle.replace("\n", "<br>")
        subtitle_html = (
            f"<span style='font-size:0.82em;color:{DEFAULT_PLOT_SETTINGS.muted_text_color};"
            f"font-weight:400'>{subtitle_html}</span>"
        )
        text = f"{title_html}<br>{subtitle_html}"

    fig.update_layout(
        title={
            "text": text,
            "x": x,
            "xanchor": "center",
            "y": 0.94,
            "yanchor": "top",
            "font": {
                "size": DEFAULT_PLOT_SETTINGS.title_fontsize + 2,
                "color": DEFAULT_PLOT_SETTINGS.text_color,
            },
        }
    )


def focus_colors(
    highlight_mask: pd.Series | list[bool],
    highlight_color: str | None = None,
    neutral_color: str | None = None,
) -> list[str]:
    """Return a color list with one highlight color and neutral fallback.

    :param highlight_mask: Boolean mask indicating highlighted marks.
    :type highlight_mask: pd.Series | list[bool]
    :param highlight_color: Optional override for highlight color.
    :type highlight_color: str | None
    :param neutral_color: Optional override for neutral color.
    :type neutral_color: str | None
    :return: Color values parallel to ``highlight_mask``.
    :rtype: list[str]
    """
    hi = highlight_color or DEFAULT_PLOT_SETTINGS.danger_color
    lo = neutral_color or DEFAULT_PLOT_SETTINGS.neutral_color
    return [hi if bool(flag) else lo for flag in list(highlight_mask)]


def style_plotly_xy_axes(
    fig,
    x_title: str,
    y_title: str,
    x_dtick: int | float | None = None,
    x_range: list[float] | tuple[float, float] | None = None,
) -> None:
    """Apply default x/y axis styling used across charts.

    :param fig: Target Plotly figure.
    :type fig: Any
    :param x_title: X-axis title.
    :type x_title: str
    :param y_title: Y-axis title.
    :type y_title: str
    :param x_dtick: Optional x-axis tick spacing.
    :type x_dtick: int | float | None
    :param x_range: Optional x-axis range.
    :type x_range: list[float] | tuple[float, float] | None
    :return: None.
    :rtype: None
    """
    x_kwargs = {
        "title_text": x_title,
        "showgrid": True,
        "gridcolor": DEFAULT_PLOT_SETTINGS.gridcolor,
        "zeroline": False,
    }
    if x_dtick is not None:
        x_kwargs["dtick"] = x_dtick
    if x_range is not None:
        x_kwargs["range"] = list(x_range)
    fig.update_xaxes(**x_kwargs)

    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
    )


# ---------------------------------------------------------------------------
# Generic figure helpers  (Gestalt: Similarity — one save pattern everywhere)
# ---------------------------------------------------------------------------

@contextmanager
def mpl_figure(
    out_path: str | Path,
    figsize: tuple[float, float],
    **subplots_kwargs: Any,
) -> Generator[tuple[Any, Any], None, None]:
    """Context manager that creates, styles, saves, and closes a Matplotlib figure.

    Applies ``DEFAULT_PLOT_SETTINGS.matplotlib_style`` and ``rcParams`` once,
    then yields ``(fig, axes)`` to the caller.  On exit it calls
    ``tight_layout()``, ``savefig()`` and ``close()`` automatically —
    even when an exception occurs (``finally`` guard).

    Usage::

        with mpl_figure("img/chart.png", (9, 5)) as (fig, ax):
            ax.bar(x, y)
            ax.set_title("My chart")

    :param out_path: Destination image path (parent directories are created).
    :type out_path: str | Path
    :param figsize: ``(width, height)`` in inches passed to ``plt.subplots``.
    :type figsize: tuple[float, float]
    :param subplots_kwargs: Extra keyword arguments forwarded to ``plt.subplots``
        (e.g. ``nrows=2``, ``gridspec_kw={…}``).
    :type subplots_kwargs: Any
    :return: Generator yielding ``(fig, axes)`` as returned by ``plt.subplots``.
    :rtype: Generator[tuple[Any, Any], None, None]
    """
    import matplotlib.pyplot as plt

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()
    fig, axes = plt.subplots(figsize=figsize, **subplots_kwargs)
    try:
        yield fig, axes
        fig.tight_layout()
        fig.savefig(ensure_parent_dir(out_path), dpi=DEFAULT_PLOT_SETTINGS.dpi)
    finally:
        plt.close(fig)


def save_plotly_fig(fig: Any, out_path: str | Path, scale: int = 2) -> None:
    """Save a Plotly figure to disk as a raster image.

    Ensures the parent directory exists before writing.

    :param fig: Plotly figure object to export.
    :type fig: Any
    :param out_path: Destination image path.
    :type out_path: str | Path
    :param scale: Pixel-density multiplier (default 2 for retina-quality output).
    :type scale: int
    :return: None.
    :rtype: None
    """
    fig.write_image(str(ensure_parent_dir(out_path)), scale=scale)
