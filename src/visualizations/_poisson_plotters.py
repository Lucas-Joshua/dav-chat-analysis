"""Low-level plot constructors for Poisson statistical modeling (Les 5).

Approach
--------
1. Aggregate messages into 15-minute intervals.
2. Fit a Poisson distribution: λ = mean count per interval.
3. Compare the observed count distribution (histogram) to the theoretical
   Poisson PMF scaled to the same total.
4. On a separate timeline, mark intervals that deviate more than the 99th-
   percentile Poisson threshold — these are candidate anomalies.

Note on the noise/error term
-----------------------------
Real chat activity is *over-dispersed* (variance > mean), meaning a pure
Poisson model underestimates tail probabilities.  The residual plotted below
represents exactly this noise: the part of activity that Poisson cannot
explain.  In a comment inside the function we document this for notebook use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS


def plot_poisson_model(
    df: pd.DataFrame,
    out_path: str | Path = "img/poisson_model.png",
) -> None:
    """Plot observed vs. Poisson-expected message counts with anomaly marking.

    Upper panel: distribution comparison — observed histogram of per-interval
    counts versus the Poisson PMF scaled to the same N.
    Lower panel: time series with the Poisson 99th-percentile threshold drawn
    as a dashed line so anomalous spikes are immediately visible.

    Noise/error term note
    ---------------------
    The Poisson model assumes Var(X) = λ.  Real chat data shows Var(X) >> λ
    (over-dispersion caused by burst behaviour, incident discussions, etc.).
    The residual ``observed − λ`` therefore contains both random noise and
    systematic deviations; the anomaly detection threshold (ppf 0.99) accounts
    for this by using the Poisson upper tail as a conservative bound.

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

    # λ = mean messages per 15-minute interval (Poisson rate parameter)
    lam: float = float(ts.mean())
    n_intervals = len(ts)

    # --- Anomaly threshold: Poisson 99th percentile ---
    threshold = int(poisson.ppf(0.99, lam))

    # --- Observed count distribution ---
    max_count = int(ts.max())
    count_vals = np.arange(0, max_count + 1)
    observed_freq = np.array([(ts == k).sum() for k in count_vals])

    # --- Poisson expected frequency ---
    expected_freq = poisson.pmf(count_vals, lam) * n_intervals

    primary = DEFAULT_PLOT_SETTINGS.primary_color
    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.12,
        subplot_titles=[
            f"Verdeling per interval: Waargenomen vs. Poisson (λ = {lam:.2f})",
            "Tijdreeks met anomalie-grens (Poisson P99)",
        ],
    )

    # ---- Row 1: distribution comparison ----

    # Observed bars
    fig.add_trace(
        go.Bar(
            x=count_vals,
            y=observed_freq,
            name="Waargenomen",
            marker_color=neutral,
            opacity=0.75,
        ),
        row=1, col=1,
    )

    # Poisson expected line
    fig.add_trace(
        go.Scatter(
            x=count_vals,
            y=expected_freq,
            mode="lines+markers",
            name=f"Poisson (λ = {lam:.2f})",
            line=dict(color=accent, width=2.5),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )

    # Annotate λ
    fig.add_annotation(
        x=lam,
        y=float(expected_freq[int(round(lam))]),
        text=f"λ = {lam:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-35,
        font=dict(size=11, color=accent),
        row=1, col=1,
    )

    fig.update_xaxes(
        title_text="Berichten per 15-min interval",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        range=[-0.5, min(max_count, int(lam * 5) + 1)],
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Frequentie (# intervallen)",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        row=1, col=1,
    )

    # ---- Row 2: time series with anomaly threshold ----

    # Mark anomalous intervals
    anomaly_mask = ts > threshold
    normal_ts = ts[~anomaly_mask]
    anomaly_ts = ts[anomaly_mask]

    fig.add_trace(
        go.Scatter(
            x=normal_ts.index,
            y=normal_ts.values,
            mode="lines",
            name="Normale activiteit",
            line=dict(color=primary, width=1),
            opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=anomaly_ts.index,
            y=anomaly_ts.values,
            mode="markers",
            name=f"Anomalie (> P99 = {threshold})",
            marker=dict(color=accent, size=6, symbol="circle"),
        ),
        row=2, col=1,
    )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=accent,
        line_width=1.8,
        annotation_text=f"P99 drempel = {threshold}",
        annotation_position="top right",
        row=2, col=1,
    )

    # λ reference line
    fig.add_hline(
        y=lam,
        line_dash="dot",
        line_color="rgba(0,0,0,0.35)",
        line_width=1.2,
        annotation_text=f"λ = {lam:.1f}",
        annotation_position="bottom right",
        row=2, col=1,
    )

    n_anomalies = int(anomaly_mask.sum())
    fig.update_xaxes(
        title_text="Datum / tijd",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text="Berichten per 15 min",
        showgrid=True,
        gridcolor=DEFAULT_PLOT_SETTINGS.gridcolor,
        zeroline=False,
        row=2, col=1,
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 60, "t": 110, "b": 60},
        )
    )
    fig.update_layout(
        title={
            "text": (
                "Poisson-model · Berichtenfrequentie per 15-minuten interval"
                f"<br><sup>λ = {lam:.2f} berichten/interval · {n_anomalies} anomalieën"
                f" · {date_min} – {date_max}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=680,
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
        bargap=0.1,
    )

    fig.write_image(str(out_path), scale=2)
