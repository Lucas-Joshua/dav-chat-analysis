"""Low-level plot constructors for time-series modeling visualizations (Les 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS


def plot_time_series_activity(
    df: pd.DataFrame,
    out_path: str | Path = "img/time_series_activity.png",
) -> None:
    """Plot 15-minute-interval chat activity with rolling trend and residuals.

    Upper panel shows raw 15-minute message counts (transparent bars) overlaid
    with a 2-hour rolling-average trend line.  The highest single interval is
    annotated as a possible incident window.  The lower panel shows the residual
    (actual − trend) so brief spikes stand out clearly.

    :param df: Processed chat dataframe with a ``datetime`` column.
    :type df: pd.DataFrame
    :param out_path: Destination path for the exported image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found in dataframe.")

    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")

    # --- Aggregate to 15-minute intervals ---
    ts: pd.Series = (
        df.set_index("datetime")
        .resample("15min")
        .size()
        .rename("count")
    )

    # --- 2-hour rolling average (8 × 15-min windows) ---
    rolling = ts.rolling(window=8, center=True, min_periods=1).mean()
    residuals = ts - rolling

    # --- Find actual peak interval ---
    peak_dt = ts.idxmax()
    peak_val = int(ts.max())

    primary = DEFAULT_PLOT_SETTINGS.primary_color
    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "15-minuten berichtenactiviteit + rollend gemiddelde (2u)",
            "Residu (Werkelijk − Trend)",
        ],
    )

    # Raw counts as faint bars
    fig.add_trace(
        go.Bar(
            x=ts.index,
            y=ts.values,
            name="Berichten (15 min)",
            marker_color=neutral,
            opacity=0.45,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Rolling average trend
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name="Rollend gemiddelde (2u)",
            line=dict(color=primary, width=2.5),
        ),
        row=1, col=1,
    )

    # Peak annotation
    fig.add_annotation(
        x=peak_dt,
        y=peak_val,
        text=f"Piek: {peak_dt.strftime('%d %b %Y %H:%M')} ({peak_val} berichten)",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-50,
        font=dict(size=10),
        row=1, col=1,
    )

    # Residuals
    res_colors = [accent if r > 0 else neutral for r in residuals.values]
    fig.add_trace(
        go.Bar(
            x=residuals.index,
            y=residuals.values,
            name="Residu",
            marker_color=res_colors,
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=0,
        line_width=1,
        line_color="rgba(0,0,0,0.3)",
        row=2, col=1,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        title_text="Datum / tijd",
        row=2, col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        title_text="Berichten",
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        title_text="Residu",
        row=2, col=1,
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 40, "t": 100, "b": 60},
        )
    )
    fig.update_layout(
        title={
            "text": (
                "Chat-activiteit over tijd · 15-minuten tijdreeks"
                f"<br><sup>Berichten per 15 min · {date_min} – {date_max}"
                " · Trend via rollend gemiddelde (2u)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=560,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        bargap=0,
    )

    fig.write_image(str(out_path), scale=2)


def plot_autocorrelation(
    df: pd.DataFrame,
    out_path: str | Path = "img/time_series_autocorrelation.png",
    max_lag: int = 96,
) -> None:
    """Plot autocorrelation function of 15-minute message counts.

    Bars are coloured red when they exceed the 95 % significance threshold
    (±1.96 / √n), revealing statistically meaningful periodic structure.
    A max_lag of 96 covers exactly 24 hours at 15-minute resolution.

    :param df: Processed chat dataframe with a ``datetime`` column.
    :type df: pd.DataFrame
    :param out_path: Destination path for the exported image.
    :type out_path: str | Path
    :param max_lag: Maximum lag to display (number of 15-minute intervals).
    :type max_lag: int
    :return: None.
    :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found in dataframe.")

    ts: pd.Series = (
        df.set_index("datetime")
        .resample("15min")
        .size()
        .rename("count")
    )

    counts = ts.values.astype(float)
    n = len(counts)
    centered = counts - counts.mean()

    # Normalized autocorrelation via full convolution
    full_acf = np.correlate(centered, centered, mode="full")
    acf = full_acf[n - 1:]          # Positive lags only
    acf = acf / acf[0]              # Normalize to 1 at lag 0
    acf = acf[1 : max_lag + 1]      # Skip lag 0; keep up to max_lag

    lags = np.arange(1, max_lag + 1)
    lag_hours = lags * 15 / 60      # Convert to hours

    sig_bound = 1.96 / np.sqrt(n)
    bar_colors = [
        DEFAULT_PLOT_SETTINGS.danger_color if abs(a) > sig_bound
        else DEFAULT_PLOT_SETTINGS.neutral_color
        for a in acf
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lag_hours,
            y=acf,
            marker_color=bar_colors,
            showlegend=False,
        )
    )

    # Significance bounds
    fig.add_hline(
        y=sig_bound,
        line_dash="dash",
        line_color="rgba(0,0,0,0.45)",
        line_width=1.4,
        annotation_text="95% grens",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-sig_bound,
        line_dash="dash",
        line_color="rgba(0,0,0,0.45)",
        line_width=1.4,
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.2)")

    fig.update_xaxes(
        title_text="Vertraging (uren)",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        tickvals=list(range(0, 25, 4)),
        ticktext=[f"{h}u" for h in range(0, 25, 4)],
    )
    fig.update_yaxes(
        title_text="Autocorrelatie",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        range=[-0.65, 1.05],
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 60, "t": 90, "b": 60},
        )
    )
    fig.update_layout(
        title={
            "text": (
                "Temporele afhankelijkheid · Autocorrelatie berichtenactiviteit"
                "<br><sup>15-minuten-intervallen · Vertraging 0 – 24 uur"
                " · Rood = statistisch significant (95%)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=420,
    )

    fig.write_image(str(out_path), scale=2)
