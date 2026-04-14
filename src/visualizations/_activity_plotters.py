"""Low-level plot constructors for chat activity visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import (
    ensure_parent_dir,
    focus_colors,
    hourly_message_counts,
    set_plotly_title,
    style_plotly_xy_axes,
)


# ---------------------------------------------------------------------------
# Public plotters
# ---------------------------------------------------------------------------

def plot_chat_activity_by_hour(
    df: pd.DataFrame,
    out_path: str | Path = "img/chat_activity_by_hour.png",
) -> None:
    """Plot hourly chat activity with one clear message: when chat is most active.

    :param df: Input dataframe with ``datetime`` and ``hour`` columns.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = ensure_parent_dir(out_path)

    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")

    hourly_counts = hourly_message_counts(df, hour_col="hour")

    peak_idx = int(hourly_counts["count"].idxmax())
    peak_hour = int(hourly_counts.loc[peak_idx, "hour"])
    peak_count = int(hourly_counts.loc[peak_idx, "count"])
    total_count = int(hourly_counts["count"].sum())
    peak_share = peak_count / max(float(total_count), 1.0)
    peak_pct = peak_share * 100.0
    peak_count_text = f"{peak_count:,}".replace(",", ".")
    bar_colors = focus_colors(hourly_counts["hour"].eq(peak_hour))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly_counts["hour"],
            y=hourly_counts["count"],
            name="Berichten per uur",
            marker_color=bar_colors,
            opacity=0.9,
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=peak_hour,
        y=peak_count,
        text=f"Piek: {peak_hour}:00 ({peak_count_text} berichten, ≈{peak_pct:.1f}% van totaal)",
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=-35,
        font=dict(size=10),
    )

    style_plotly_xy_axes(
        fig,
        x_title="Uur van de dag",
        y_title="Aantal berichten",
        x_dtick=2,
        x_range=(-0.5, 23.5),
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 40, "t": 62, "b": 60},
        )
    )
    set_plotly_title(
        fig,
        title="Chatactiviteit piekt in de vroege avond",
        subtitle=(
            f"{date_min} – {date_max} · "
            f"rood = piekuur ({peak_hour}:00) · "
            f"≈{peak_pct:.1f}% van alle berichten"
        ),
    )
    fig.update_layout(height=480)

    fig.write_image(out_path, scale=2)

def plot_chat_activity_weekday_weekend(
    df: pd.DataFrame,
    out_path: str | Path = "img/chat_activity_weekday_weekend.png",
) -> None:
    """Compare daily activity distribution between weekdays and weekend.

    Uses a dot/strip plot so each day is visible as an individual observation.
    Median lines per group are drawn as subtle dashed rules.

    :param df: Input dataframe containing ``datetime``.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")

    out_path = ensure_parent_dir(out_path)
    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])
    working["date_only"] = working["datetime"].dt.date
    working["is_weekend"] = working["datetime"].dt.weekday >= 5

    daily = (
        working.groupby(["date_only", "is_weekend"])
        .size()
        .reset_index(name="messages")
    )
    weekday = daily.loc[~daily["is_weekend"], "messages"].astype(float).to_numpy()
    weekend = daily.loc[daily["is_weekend"], "messages"].astype(float).to_numpy()
    if len(weekday) == 0 or len(weekend) == 0:
        raise ValueError("Need both weekday and weekend data for comparison.")

    color_weekday = DEFAULT_PLOT_SETTINGS.neutral_color  # "#B0B0B0"
    color_weekend = DEFAULT_PLOT_SETTINGS.danger_color   # "#C62828"

    median_weekday = float(np.median(weekday))
    median_weekend = float(np.median(weekend))

    rng = np.random.default_rng(42)

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # --- Dot strip for weekdays (x-position 1) ---
    jitter_wd = rng.uniform(-0.22, 0.22, size=len(weekday))
    ax.scatter(
        1 + jitter_wd,
        weekday,
        s=48,
        color=color_weekday,
        alpha=0.60,
        linewidths=0,
        zorder=3,
        label="Weekdag",
    )

    # --- Dot strip for weekend (x-position 2) ---
    jitter_we = rng.uniform(-0.22, 0.22, size=len(weekend))
    ax.scatter(
        2 + jitter_we,
        weekend,
        s=48,
        color=color_weekend,
        alpha=0.65,
        linewidths=0,
        zorder=3,
        label="Weekend",
    )

    # --- Median lines (solid, prominent) ---
    half_width = 0.32
    ax.hlines(
        median_weekday,
        1 - half_width,
        1 + half_width,
        colors=DEFAULT_PLOT_SETTINGS.text_color,
        linewidths=2.0,
        linestyles="solid",
        zorder=5,
    )
    ax.hlines(
        median_weekend,
        2 - half_width,
        2 + half_width,
        colors=DEFAULT_PLOT_SETTINGS.text_color,
        linewidths=2.0,
        linestyles="solid",
        zorder=5,
    )

    # --- IQR band per group (shaded range between P25 and P75) ---
    for x_pos, values, color in [
        (1, weekday, color_weekday),
        (2, weekend, color_weekend),
    ]:
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        ax.fill_betweenx(
            [q25, q75],
            x_pos - 0.26,
            x_pos + 0.26,
            color=color,
            alpha=0.14,
            zorder=1,
        )

    # --- Annotation: point upward to spread in weekend upper cloud ---
    q75_we = float(np.percentile(weekend, 75))
    ax.annotate(
        "Weekend: bredere spreiding\nin de middelste massa",
        xy=(2.24, q75_we + 4),
        xytext=(2.56, q75_we + 18),
        fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize,
        color=color_weekend,
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.12",
            color=color_weekend,
            lw=1.0,
        ),
        va="bottom",
        ha="left",
        bbox=DEFAULT_PLOT_SETTINGS.annotation_box,
    )

    # --- Axes & labels ---
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Weekdagen", "Weekend"], fontsize=DEFAULT_PLOT_SETTINGS.axis_label_fontsize)
    ax.set_xlim(0.4, 3.1)
    ax.set_ylabel("Berichten per dag")
    ax.set_title("Weekend laat grotere spreiding in chatactiviteit zien")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Sample-size footnote
    ax.text(
        0.98,
        0.98,
        f"n weekdagen={len(weekday)} · n weekend={len(weekend)}  |  lijn = mediaan  |  vlak = IQR",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)
