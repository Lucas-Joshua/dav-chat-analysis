"""Low-level plot constructors for time-series modeling visualizations (Les 3)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.modules.feature_engineering import INCIDENT_BOW_TERMS
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS

_GRID_COLOR = "#D9D9D9"


def _get_incident_windows(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the 15-min window timestamps that contain ≥1 incident message.

    :param df: Dataframe with ``datetime`` and ``message`` columns.
    :type df: pd.DataFrame
    :return: DatetimeIndex of incident-window floor timestamps (15-min grid).
    :rtype: pd.DatetimeIndex
    """
    terms = [str(t) for t in INCIDENT_BOW_TERMS]
    pattern = r"\b(?:{})\b".format(
        "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in terms)
    )
    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])
    working["is_incident"] = (
        working["message"]
        .fillna("")
        .astype(str)
        .str.contains(pattern, case=False, regex=True)
    )
    incident_rows = working[working["is_incident"]].copy()
    incident_rows["window"] = incident_rows["datetime"].dt.floor("15min")
    return pd.DatetimeIndex(incident_rows["window"].unique())


def plot_time_series_activity(
    df: pd.DataFrame,
    out_path: str | Path = "img/time_series_activity.png",
) -> None:
    """Plot 15-minute chat activity with incident markers and Poisson λ annotation.

    Upper panel shows raw 15-minute message counts (grey bars) with a 2-hour
    rolling-average trend line (blue) and vertical red markers at every window
    that contains ≥1 incident-related message.  The lower panel shows residuals
    (actual − trend) so brief incident spikes stand out clearly.

    Title is insight-driven: activity measurably increases during incident windows.

    :param df: Processed chat dataframe with ``datetime`` and ``message`` columns.
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

    # --- Detect incident windows ---
    incident_windows = _get_incident_windows(df)

    # λ per incident window vs baseline
    incident_mask = ts.index.isin(incident_windows)
    lam_incident = float(ts[incident_mask].mean()) if incident_mask.any() else 0.0
    lam_baseline = float(ts[~incident_mask & (ts > 0)].mean())

    # --- Find peak interval ---
    peak_dt = ts.idxmax()
    peak_val = int(ts.max())

    import matplotlib.patches as mpatches

    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color
    primary = DEFAULT_PLOT_SETTINGS.primary_color

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2.2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    ax_top.set_facecolor("white")
    ax_bot.set_facecolor("white")

    # --- Top: raw bars + rolling trend ---
    ax_top.bar(ts.index, ts.values, width=0.009, color=neutral, alpha=0.40,
               label="Berichten per 15 min")
    ax_top.plot(rolling.index, rolling.values, color=primary, linewidth=1.8,
                label="Rollend gemiddelde (2u)")

    # --- Incident: filled spans — much more visible than dotted lines ---
    _span_w = pd.Timedelta("15min")
    for inc_dt in incident_windows:
        ax_top.axvspan(inc_dt, inc_dt + _span_w, color=accent, alpha=0.22, linewidth=0)
    incident_patch = mpatches.Patch(
        color=accent, alpha=0.55,
        label=f"Incident-venster (n={len(incident_windows)})",
    )

    # --- Peak annotation ---
    ax_top.annotate(
        f"Piek {peak_dt.strftime('%d %b %H:%M')}\n({peak_val} berichten)",
        xy=(peak_dt, peak_val),
        xytext=(0, 18), textcoords="offset points",
        ha="center", fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="->,head_width=0.25", color="#555555", lw=0.9),
    )

    # --- λ-annotation box (upper right, inside plot) incl. date range ---
    lambda_text = (
        f"m\u209c ~ Pois(\u03bb)  \u00b7  {date_min}\u2013{date_max}\n"
        f"\u03bb normaal\u202f=\u202f{lam_baseline:.2f} bericht/15min\n"
        f"\u03bb incident\u202f=\u202f{lam_incident:.2f} bericht/15min"
    )
    ax_top.text(
        0.995, 0.97, lambda_text,
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=9.5, color=DEFAULT_PLOT_SETTINGS.text_color,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#CCCCCC", lw=0.9),
        linespacing=1.55,
        family="monospace",
    )

    ax_top.set_ylabel("Berichten per 15 min", fontsize=11)
    ax_top.set_title(
        "Chatactiviteit piekt zichtbaar bij incidenten",
        fontsize=12, fontweight="bold", pad=12,
    )
    handles, labels_leg = ax_top.get_legend_handles_labels()
    handles.append(incident_patch)
    labels_leg.append(incident_patch.get_label())
    ax_top.legend(handles=handles, labels=labels_leg,
                  fontsize=9, frameon=False, loc="upper left")
    ax_top.grid(axis="y", alpha=0.18, color=_GRID_COLOR)
    ax_top.grid(axis="x", visible=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- Bottom: residuals ---
    res_vals = residuals.values
    bar_colors_res = [accent if r > 0 else neutral for r in res_vals]
    ax_bot.bar(residuals.index, res_vals, width=0.009, color=bar_colors_res, alpha=0.75)
    ax_bot.axhline(0, color="#333333", linewidth=0.8)
    ax_bot.set_ylabel("Residu\n(werkelijk \u2212 trend)", fontsize=9)
    ax_bot.set_xlabel("Datum / tijd", fontsize=11)
    ax_bot.grid(axis="y", alpha=0.15, color=_GRID_COLOR)
    ax_bot.grid(axis="x", visible=False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


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
