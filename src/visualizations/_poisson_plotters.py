"""Low-level plot constructors for Poisson distribution comparison (Les 4 DIST)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson

from src.modules.feature_engineering import INCIDENT_BOW_TERMS
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import ensure_parent_dir, set_plotly_title, style_plotly_xy_axes


def _classify_days(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split daily message counts into normal days and incident days.

    A day is classified as an 'incident day' when it contains ≥1 message that
    matches the incident bag-of-words.  Daily aggregation gives enough data per
    group for a stable Poisson fit and makes the λ-shift clearly visible.

    :param df: Dataframe with ``datetime`` and ``message`` columns.
    :type df: pd.DataFrame
    :return: Tuple (normal_counts, incident_counts) as pandas Series.
    :rtype: tuple[pd.Series, pd.Series]
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

    daily_total: pd.Series = (
        working.set_index("datetime").resample("1D").size().rename("count")
    )
    daily_flag: pd.Series = (
        working.set_index("datetime")["is_incident"]
        .resample("1D")
        .sum()
        .gt(0)
    )

    active = daily_total > 0  # only days with ≥1 message
    incident_counts = daily_total[daily_flag & active]
    normal_counts = daily_total[~daily_flag & active]
    return normal_counts, incident_counts


def plot_poisson_dual_distribution(
    df: pd.DataFrame,
    out_path: str | Path = "img/poisson_dual_distribution.png",
) -> None:
    """Compare daily message-count distributions for normal vs incident days.

    Two overlapping histograms (grey = normal, red = incident days) are shown
    with their fitted Poisson PMF curves, making the lambda-shift immediately
    visible.  Models chat as a Poisson counting process where incident days
    correspond to a higher intensity parameter (lambda_1 > lambda_0).

    :param df: Processed chat dataframe with ``datetime`` and ``message`` columns.
    :type df: pd.DataFrame
    :param out_path: Destination path for the exported image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normal_counts, incident_counts = _classify_days(df)

    lam0 = float(normal_counts.mean())
    lam1 = float(incident_counts.mean()) if len(incident_counts) > 0 else 0.0
    n_normal = len(normal_counts)
    n_incident = len(incident_counts)

    color_normal = DEFAULT_PLOT_SETTINGS.neutral_color
    color_incident = DEFAULT_PLOT_SETTINGS.danger_color

    max_val = int(max(normal_counts.max(), incident_counts.max() if n_incident else 0))
    # Cap x-range so the plot stays readable
    x_max_pmf = min(max_val + 2, int(lam1 * 3 + 20))
    x_pmf = np.arange(0, x_max_pmf + 1)

    pmf0 = poisson.pmf(x_pmf, lam0) * n_normal
    pmf1 = poisson.pmf(x_pmf, lam1) * n_incident if n_incident > 0 else np.zeros_like(x_pmf, dtype=float)

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Bin edges covering both distributions
    bins = np.arange(0, min(max_val + 3, x_max_pmf + 3)) - 0.5

    ax.hist(
        normal_counts,
        bins=bins,
        color=color_normal,
        alpha=0.55,
        label=f"Normale dag (n={n_normal}, lam_0={lam0:.1f})",
        density=False,
    )
    if n_incident > 0:
        ax.hist(
            incident_counts,
            bins=bins,
            color=color_incident,
            alpha=0.65,
            label=f"Incident-dag (n={n_incident}, lam_1={lam1:.1f})",
            density=False,
        )

    # Poisson fit curves
    ax.plot(x_pmf, pmf0, color="#555555", linewidth=1.8, linestyle="--",
            label=f"Pois(lam_0={lam0:.1f})")
    if n_incident > 0:
        ax.plot(x_pmf, pmf1, color=color_incident, linewidth=2.0, linestyle="--",
                label=f"Pois(lam_1={lam1:.1f})")

    # lambda-shift annotation — point to where the incident PMF peaks
    if n_incident > 0:
        peak_x = int(round(lam1))
        peak_y = float(pmf1[peak_x]) if peak_x < len(pmf1) else float(pmf1[-1])
        pct = (lam1 / lam0 - 1) * 100
        ax.annotate(
            f"Delta-lam = {lam1 - lam0:+.1f}\n(+{pct:.0f}% intensiteit)",
            xy=(peak_x, peak_y),
            xytext=(peak_x + max(4, int(lam1 * 0.4)), peak_y * 1.4),
            fontsize=10,
            color=color_incident,
            arrowprops=dict(arrowstyle="->,head_width=0.3", color=color_incident, lw=1.0),
        )

    ax.set_xlabel("Berichten per dag", fontsize=11)
    ax.set_ylabel("Aantal dagen", fontsize=11)
    ax.set_title(
        "Incident-dagen volgen een hogere Poisson-intensiteit (lam_1 > lam_0)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, color="#D9D9D9")
    ax.grid(axis="x", visible=False)

    ax.text(
        0.98, 0.98,
        "m_dag ~ Pois(lam)  |  incident-dag = hogere intensiteit lam",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color=DEFAULT_PLOT_SETTINGS.muted_text_color,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def plot_poisson_model(
    df: pd.DataFrame,
    out_path: str | Path = "img/poisson_model.png",
) -> None:
    """Plot observed frequency versus Poisson expectation in one clear chart.

    :param df: Processed chat dataframe with a ``datetime`` column.
    :type df: pd.DataFrame
    :param out_path: Destination path for the exported image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    out_path = ensure_parent_dir(out_path)

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found in dataframe.")

    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")

    # --- Aggregate to an interpretable interval ---
    ts_hourly: pd.Series = (
        df.set_index("datetime")
        .resample("1h")
        .size()
        .rename("count")
    )
    if float(ts_hourly.mean()) < 2.0:
        ts = (
            df.set_index("datetime")
            .resample("1d")
            .size()
            .rename("count")
        )
        interval_label = "dag"
    else:
        ts = ts_hourly
        interval_label = "uur"

    # λ = mean messages per gekozen interval (Poisson rate parameter)
    lam: float = float(ts.mean())
    n_intervals = len(ts)

    # --- Observed count distribution ---
    max_count = int(ts.max())
    count_vals = np.arange(0, max_count + 1)
    observed_freq = np.array([(ts == k).sum() for k in count_vals])

    # --- Poisson expected frequency ---
    expected_freq = poisson.pmf(count_vals, lam) * n_intervals
    threshold = int(poisson.ppf(0.99, lam))
    outlier_share = float((ts > threshold).mean())

    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=count_vals,
            y=observed_freq,
            name="Waargenomen",
            marker_color=neutral,
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=count_vals,
            y=expected_freq,
            mode="lines+markers",
            name=f"Poisson (λ = {lam:.2f})",
            line=dict(color=accent, width=2.5),
            marker=dict(size=5),
        )
    )

    fig.add_annotation(
        x=lam,
        y=float(expected_freq[int(round(lam))]),
        text=f"λ = {lam:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-35,
        font=dict(size=11, color=accent),
    )

    style_plotly_xy_axes(
        fig,
        x_title=f"Berichten per {interval_label}-interval",
        y_title="Frequentie (# intervallen)",
        x_range=(-0.5, min(max_count, int(lam * 5) + 1)),
    )

    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 40, "t": 100, "b": 60},
        )
    )
    set_plotly_title(
        fig,
        title=f"Verdeling van chatactiviteit per {interval_label}",
        subtitle=f"Meeste intervallen hebben lage aantallen; uitschieters (>P99={threshold}) ≈ {outlier_share:.1%}",
    )
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        bargap=0.1,
    )

    fig.write_image(out_path, scale=2)
