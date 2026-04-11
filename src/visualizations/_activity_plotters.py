"""Low-level plot constructors for chat activity visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_model(t: np.ndarray, amplitude: float, phase: float, offset: float) -> np.ndarray:
    """Sinusoidal model with 24-hour period.

    :param t: Time values (hours 0–23).
    :type t: np.ndarray
    :param amplitude: Wave amplitude.
    :type amplitude: float
    :param phase: Phase offset in radians.
    :type phase: float
    :param offset: Vertical offset (mean level).
    :type offset: float
    :return: Modelled values at each time point.
    :rtype: np.ndarray
    """
    return amplitude * np.sin(2 * np.pi * t / 24 + phase) + offset


# ---------------------------------------------------------------------------
# Public plotters
# ---------------------------------------------------------------------------

def plot_chat_activity_by_hour(
    df: pd.DataFrame,
    out_path: str | Path = "img/chat_activity_by_hour.png",
) -> None:
    """Plot hourly chat activity with cubic-spline smoothing and sine decomposition.

    The top panel shows the smoothed actual activity overlaid with a fitted
    sinusoidal trend.  The bottom panel shows the residual (actual minus sine)
    so deviations from the daily rhythm are immediately visible.  The real
    activity peak is annotated automatically.

    :param df: Input dataframe with ``datetime`` and ``hour`` columns.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "hour" not in df.columns:
        raise KeyError("hour column not found. Run add_time_features first.")

    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")

    hourly_counts = (
        df["hour"]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
        .reset_index()
    )
    hourly_counts.columns = ["hour", "count"]

    hours = hourly_counts["hour"].to_numpy(dtype=float)
    counts = hourly_counts["count"].to_numpy(dtype=float)

    # --- Cubic-spline smooth (more interpolated points → smoother line) ---
    cs = CubicSpline(hours, counts)
    t_fine = np.linspace(0, 23, 300)
    y_smooth = np.clip(cs(t_fine), 0, None)

    # --- Sine-wave fit ---
    try:
        p0 = [max(counts.std(), 1.0), 0.0, counts.mean()]
        popt, _ = curve_fit(_sine_model, hours, counts, p0=p0, maxfev=8000)
        y_sine = _sine_model(t_fine, *popt)
        y_sine_at_hours = _sine_model(hours, *popt)
        residuals = counts - y_sine_at_hours
        sine_ok = True
    except RuntimeError:
        sine_ok = False

    # --- Actual peak (not hardcoded) ---
    peak_idx = int(np.argmax(counts))
    peak_hour = int(hours[peak_idx])
    peak_count = int(counts[peak_idx])

    primary = DEFAULT_PLOT_SETTINGS.primary_color
    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color

    # --- Build figure ---
    if sine_ok:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.68, 0.32],
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=["Activiteit per uur + sinusfit", "Residu (Werkelijk − Sinus)"],
        )

        # Smoothed actual line
        fig.add_trace(
            go.Scatter(
                x=t_fine,
                y=y_smooth,
                mode="lines",
                name="Berichten (smooth)",
                line=dict(color=primary, width=2.5),
            ),
            row=1, col=1,
        )

        # Original hourly dots
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=counts,
                mode="markers",
                name="Uurgemiddelde",
                marker=dict(color=primary, size=6, opacity=0.65),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Fitted sine
        fig.add_trace(
            go.Scatter(
                x=t_fine,
                y=y_sine,
                mode="lines",
                name="Sinusfit",
                line=dict(color=accent, width=2, dash="dot"),
            ),
            row=1, col=1,
        )

        # Peak annotation
        fig.add_annotation(
            x=peak_hour,
            y=peak_count,
            text=f"Piek: {peak_hour}:00 uur ({peak_count:,} berichten)",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-45,
            font=dict(size=11),
            row=1, col=1,
        )

        # Residuals bar chart
        res_colors = [
            accent if r > 0 else neutral for r in residuals
        ]
        fig.add_trace(
            go.Bar(
                x=hours,
                y=residuals,
                name="Residu",
                marker_color=res_colors,
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0,
            line_width=1,
            line_dash="solid",
            line_color="rgba(0,0,0,0.3)",
            row=2, col=1,
        )

        fig.update_xaxes(
            range=[-0.5, 23.5],
            tickmode="linear",
            dtick=2,
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            title_text="Uur van de dag",
            row=2, col=1,
        )
        fig.update_xaxes(
            range=[-0.5, 23.5],
            tickmode="linear",
            dtick=2,
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            row=1, col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            title_text="Aantal berichten",
            row=1, col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            title_text="Residu",
            row=2, col=1,
        )

        height = 580

    else:
        # Fallback when sine fit fails: single-panel smooth line
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t_fine,
                y=y_smooth,
                mode="lines",
                name="Berichten",
                line=dict(color=primary, width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=counts,
                mode="markers",
                marker=dict(color=primary, size=6, opacity=0.65),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=peak_hour,
            y=peak_count,
            text=f"Piek: {peak_hour}:00 uur ({peak_count:,} berichten)",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-45,
        )
        fig.update_xaxes(
            range=[-0.5, 23.5],
            tickmode="linear",
            dtick=2,
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            title_text="Uur van de dag",
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
            zeroline=False,
            title_text="Aantal berichten",
        )
        height = 420

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 40, "t": 100, "b": 60},
        )
    )
    fig.update_layout(
        title={
            "text": (
                "Chat-activiteit per uur van de dag · Sinus-decompositie"
                f"<br><sup>Berichten per uur · {date_min} – {date_max}"
                " · Alle dagen geaggregeerd · Cubic-spline smoothing</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=height,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )

    fig.write_image(str(out_path), scale=2)


def plot_chat_activity_distribution(
    df: pd.DataFrame,
    output: Optional[Path] = None,
) -> go.Figure:
    """Visualize hourly chat-message distribution with a KDE curve overlay.

    A bar chart shows raw message counts per hour.  A Gaussian KDE curve is
    fitted to the same data (scaled to message-count units) and overlaid so
    the underlying continuous distribution is visible next to the discrete
    bars.

    :param df: Input dataframe with an ``hour`` column.
    :type df: pd.DataFrame
    :param output: Optional output image path.
    :type output: Optional[Path]
    :return: Plotly figure with hourly distribution bars and KDE overlay.
    :rtype: go.Figure
    """
    df = df.copy()

    hourly_counts = (
        df.groupby("hour")
        .size()
        .reset_index(name="messages")
        .sort_values("hour")
    )

    primary = DEFAULT_PLOT_SETTINGS.primary_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color

    # --- KDE scaled to total message count ---
    hours_expanded = np.repeat(
        hourly_counts["hour"].values,
        hourly_counts["messages"].values,
    )
    kde = gaussian_kde(hours_expanded, bw_method=0.25)
    x_fine = np.linspace(0, 23, 300)
    kde_scaled = kde(x_fine) * hourly_counts["messages"].sum()

    # --- Figure ---
    fig = go.Figure()

    # Bars
    fig.add_trace(
        go.Bar(
            x=hourly_counts["hour"],
            y=hourly_counts["messages"],
            name="Berichten per uur",
            marker_color=neutral,
            showlegend=False,
        )
    )

    # KDE overlay
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=kde_scaled,
            mode="lines",
            name="Kansdichtheid (KDE)",
            line=dict(color=primary, width=2.5),
        )
    )

    fig.update_layout(
        template=DEFAULT_PLOT_SETTINGS.plotly_template,
        bargap=0.35,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        title={
            "text": (
                "Meeste chat-activiteit tijdens operationele uren van de dropzone"
                "<br><sup>Verdeling van berichten over de dag (Oct 2024 – Feb 2026)"
                " · staafdiagram + KDE-curve</sup>"
            ),
            "x": 0.5,
        },
    )
    fig.update_xaxes(
        dtick=3,
        title_text="Uur van de dag",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        title_text="Aantal berichten",
    )

    if output:
        fig.write_image(str(output))

    return fig
