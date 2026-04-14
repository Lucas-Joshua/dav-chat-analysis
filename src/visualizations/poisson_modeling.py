"""Poisson distribution visualizations and registry entries."""

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
from src.visualizations.utils import (
    ensure_parent_dir,
    resolve_lesson_output_path,
    set_plotly_title,
    style_plotly_xy_axes,
)


def _hourly_message_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate message counts to hourly intervals."""
    if "datetime" not in df.columns:
        raise KeyError("datetime column not found in dataframe.")

    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])
    return working.set_index("datetime").resample("1h").size().rename("count")


def _expected_frequency_from_rates(count_values: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Return expected frequency per count under a possibly varying rate."""
    return np.array([poisson.pmf(k, rates).sum() for k in count_values], dtype=float)


def _fit_error(observed: np.ndarray, expected: np.ndarray) -> float:
    """Compute a simple fit error that is easy to explain in the report."""
    return float(np.sqrt(np.mean((observed - expected) ** 2)))


def _classify_days(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split daily message counts into normal days and incident days."""
    terms = [str(t) for t in INCIDENT_BOW_TERMS]
    pattern = r"\b(?:{})\b".format("|".join(re.escape(t).replace(r"\ ", r"\s+") for t in terms))
    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])
    working["is_incident"] = (
        working["message"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
    )

    daily_total = working.set_index("datetime").resample("1D").size().rename("count")
    daily_flag = working.set_index("datetime")["is_incident"].resample("1D").sum().gt(0)
    active = daily_total > 0
    incident_counts = daily_total[daily_flag & active]
    normal_counts = daily_total[~daily_flag & active]
    return normal_counts, incident_counts


def plot_poisson_dual_distribution(
    df: pd.DataFrame,
    out_path: str | Path = "img/poisson_dual_distribution.png",
) -> None:
    """Compare daily message-count distributions for normal vs incident days."""
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
    x_max_pmf = min(max_val + 2, int(lam1 * 2.8 + 8)) if n_incident > 0 else int(lam0 * 4)
    x_max_pmf = max(x_max_pmf, int(lam0 * 3))
    x_pmf = np.arange(0, x_max_pmf + 1)

    pmf0 = poisson.pmf(x_pmf, lam0) * n_normal
    pmf1 = poisson.pmf(x_pmf, lam1) * n_incident if n_incident > 0 else np.zeros_like(x_pmf, dtype=float)

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.arange(0, x_max_pmf + 3) - 0.5

    ax.hist(normal_counts.clip(upper=x_max_pmf), bins=bins, color=color_normal, alpha=0.55, label=f"Normale dag  (n={n_normal}, λ₀≈{lam0:.1f})", density=False)
    if n_incident > 0:
        ax.hist(incident_counts.clip(upper=x_max_pmf), bins=bins, color=color_incident, alpha=0.65, label=f"Incident-dag  (n={n_incident}, λ₁≈{lam1:.1f})", density=False)

    ax.plot(x_pmf, pmf0, color="#555555", linewidth=2.0, linestyle="-", label=f"Pois(λ₀={lam0:.1f})")
    if n_incident > 0:
        ax.plot(x_pmf, pmf1, color=color_incident, linewidth=2.2, linestyle="-", label=f"Pois(λ₁={lam1:.1f})")

    if n_incident > 0:
        peak_x = int(round(lam1))
        peak_y = float(pmf1[min(peak_x, len(pmf1) - 1)])
        pct = (lam1 / lam0 - 1) * 100
        text_x = peak_x + max(6, int((x_max_pmf - peak_x) * 0.30))
        text_y = peak_y * 2.0
        ax.annotate(
            f"Δλ = {lam1 - lam0:+.1f}  (+{pct:.0f}%)\nincidentdag is 3× actiever",
            xy=(peak_x, peak_y),
            xytext=(text_x, text_y),
            fontsize=10,
            fontweight="semibold",
            color=color_incident,
            arrowprops=dict(arrowstyle="->,head_width=0.3", color=color_incident, lw=1.1, connectionstyle="arc3,rad=-0.18"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_incident, lw=0.8, alpha=0.9),
        )

    ax.set_xlabel("Berichten per dag")
    ax.set_ylabel("Aantal dagen")
    ax.set_title("Op incidentdagen is de Poisson-intensiteit meer dan 3× zo hoog")
    ax.legend(fontsize=DEFAULT_PLOT_SETTINGS.legend_fontsize, frameon=False, loc="upper right")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.text(
        0.02,
        0.98,
        "m_dag ∼ Pois(λ)  ·  alleen actieve dagen (n≥1 bericht)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def plot_poisson_model(
    df: pd.DataFrame,
    out_path: str | Path = "img/poisson_model.png",
) -> None:
    """Compare global vs time-dependent Poisson fits for hourly chat counts."""
    out_path = ensure_parent_dir(out_path)
    date_min = df["datetime"].min().strftime("%b %Y")
    date_max = df["datetime"].max().strftime("%b %Y")
    ts = _hourly_message_series(df)

    global_lambda = float(ts.mean())
    count_vals = np.arange(0, int(ts.max()) + 1)
    observed_freq = np.array([(ts == k).sum() for k in count_vals], dtype=float)

    hourly_lambda = ts.groupby(ts.index.hour).mean().reindex(range(24), fill_value=0.0)
    interval_rates = ts.index.hour.map(hourly_lambda).to_numpy(dtype=float)

    expected_global = _expected_frequency_from_rates(count_vals, np.full(len(ts), global_lambda, dtype=float))
    expected_time = _expected_frequency_from_rates(count_vals, interval_rates)

    global_rmse = _fit_error(observed_freq, expected_global)
    time_rmse = _fit_error(observed_freq, expected_time)
    rmse_gain = global_rmse - time_rmse

    threshold = int(poisson.ppf(0.99, global_lambda))
    outlier_share = float((ts > threshold).mean())
    peak_hour = int(hourly_lambda.idxmax())
    peak_lambda = float(hourly_lambda.loc[peak_hour])
    x_max = min(int(ts.max()), int(global_lambda * 5) + 2)

    accent = DEFAULT_PLOT_SETTINGS.danger_color
    neutral = DEFAULT_PLOT_SETTINGS.neutral_color
    model_color = DEFAULT_PLOT_SETTINGS.primary_color

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=count_vals,
            y=observed_freq,
            name="Waargenomen",
            marker_color=neutral,
            opacity=0.45,
            hovertemplate="Berichten per uur: %{x}<br>Waargenomen frequentie: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=count_vals,
            y=expected_global,
            mode="lines+markers",
            name="Globale Poisson",
            line=dict(color="rgba(47,109,179,0.65)", width=1.8, dash="dash"),
            marker=dict(size=3.5, color="rgba(47,109,179,0.65)"),
            hovertemplate="Globale Poisson<br>k=%{x}<br>Verwachte frequentie: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=count_vals,
            y=expected_time,
            mode="lines+markers",
            name="Tijdsafhankelijke Poisson",
            line=dict(color=accent, width=3.2),
            marker=dict(size=5.5, color=accent),
            hovertemplate="Tijdsafhankelijke Poisson<br>k=%{x}<br>Verwachte frequentie: %{y:.0f}<extra></extra>",
        )
    )

    global_idx = min(max(1, int(round(global_lambda))), len(count_vals) - 1)
    time_idx = min(max(1, int(round(peak_lambda))), len(count_vals) - 1)
    fig.add_annotation(
        x=count_vals[time_idx],
        y=float(expected_time[time_idx]),
        text=f"Tijdsafhankelijk beter<br>RMSE {time_rmse:.1f}",
        showarrow=True,
        arrowhead=2,
        ax=55,
        ay=-30,
        font=dict(size=10, color=accent),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=accent,
        borderwidth=1,
    )
    fig.add_annotation(
        x=count_vals[global_idx],
        y=float(expected_global[global_idx]),
        text=f"Globale λ<br>RMSE {global_rmse:.1f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=40,
        font=dict(size=9, color=model_color),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(47,109,179,0.65)",
        borderwidth=1,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.93,
        text=f"Verbetering: {rmse_gain:.1f} RMSE<br>Piekuur λ<sub>{peak_hour:02d}:00</sub> = {peak_lambda:.2f}",
        showarrow=False,
        align="right",
        font=dict(size=10, color=DEFAULT_PLOT_SETTINGS.text_color),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC",
        borderwidth=1,
    )

    style_plotly_xy_axes(fig, x_title="Berichten per uur-interval", y_title="Frequentie (# intervallen)", x_range=(-0.5, x_max))
    fig.update_xaxes(dtick=1, tickmode="linear")
    fig.update_layout(
        DEFAULT_PLOT_SETTINGS.base_plotly_layout(
            margin={"l": 60, "r": 40, "t": 100, "b": 60},
        )
    )
    set_plotly_title(
        fig,
        title="Uurafhankelijke Poisson past beter",
        subtitle=(
            f"{date_min} – {date_max} · per uur-interval · "
            f"globale λ vlakt dagritme te veel uit · "
            f"uitschieters (>P99={threshold}) ≈ {outlier_share:.1%}"
        ),
    )
    fig.update_layout(height=520, bargap=0.1, showlegend=False)
    fig.write_image(out_path, scale=2)


def poisson_model(df, out_dir: str | Path | None = None) -> None:
    """Generate the Poisson observed-vs-expected distribution plot."""
    plot_poisson_model(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "poisson_model",
            "poisson_model.png",
        ),
    )


REGISTRY = {"poisson_model": poisson_model}
