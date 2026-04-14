"""Low-level plot constructors for incident timeline visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re
from scipy import stats

from src.modules.feature_engineering import INCIDENT_BOW_TERMS
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS

_MATPLOTLIB_GRID_COLOR = "#D9D9D9"


def plot_incident_event_study(
    df: pd.DataFrame,
    out_path: str | Path = "img/incident_event_study.png",
    window_minutes: int = 15,
    lags: int = 8,
) -> None:
    """Show average chat activity before, during, and after incident windows.

    For each 15-min window that contains ≥1 incident-BOW message, the
    surrounding ±``lags`` windows are collected.  The average message count per
    lag position is plotted as a line with a shaded 95% CI band.  This
    visualises the coordinated group response to incidents without implying
    causation.

    :param df: Dataframe with ``datetime`` and ``message`` columns.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :param window_minutes: Resampling window in minutes (default 15).
    :type window_minutes: int
    :param lags: Number of windows before and after to include (default 8 = 2 h).
    :type lags: int
    :return: None.
    :rtype: None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = _build_incident_pattern()
    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])
    working["is_incident"] = (
        working["message"]
        .fillna("")
        .astype(str)
        .str.contains(pattern, case=False, regex=True)
    )

    freq = f"{window_minutes}min"
    ts: pd.Series = working.set_index("datetime").resample(freq).size().rename("count")
    flag: pd.Series = (
        working.set_index("datetime")["is_incident"]
        .resample(freq)
        .sum()
        .gt(0)
    )

    incident_indices = np.where(flag.values)[0]

    if len(incident_indices) == 0:
        return

    # Collect ±lags windows around each incident
    profiles: list[np.ndarray] = []
    n = len(ts)
    for idx in incident_indices:
        start = idx - lags
        end = idx + lags + 1
        if start < 0 or end > n:
            continue
        profiles.append(ts.values[start:end])

    if not profiles:
        return

    profile_arr = np.array(profiles, dtype=float)
    mean_profile = profile_arr.mean(axis=0)
    sem_profile = profile_arr.std(axis=0) / np.sqrt(len(profile_arr))
    lag_axis = np.arange(-lags, lags + 1)
    lag_hours = lag_axis * window_minutes / 60

    # Baseline = mean of non-incident windows (lag ≠ 0)
    baseline = float(ts[~flag].mean())
    pct_increase = (mean_profile[lags] - baseline) / max(baseline, 1e-9) * 100

    color_line = DEFAULT_PLOT_SETTINGS.danger_color
    color_base = DEFAULT_PLOT_SETTINGS.neutral_color

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 95% CI band — light alpha so the mean line stays dominant
    ci_lower = np.maximum(mean_profile - 1.96 * sem_profile, 0)
    ci_upper = mean_profile + 1.96 * sem_profile
    ax.fill_between(
        lag_hours, ci_lower, ci_upper,
        alpha=0.10, color=color_line,
        label="95% betrouwbaarheidsband",
    )
    # Thin dashed CI borders for reference
    ax.plot(lag_hours, ci_lower, color=color_line, linewidth=0.6, linestyle="--", alpha=0.35)
    ax.plot(lag_hours, ci_upper, color=color_line, linewidth=0.6, linestyle="--", alpha=0.35)

    # Mean profile
    ax.plot(
        lag_hours, mean_profile,
        color=color_line, linewidth=2.4,
        marker="o", markersize=5,
        label=f"Gem. berichten per {window_minutes} min  (n={len(profiles)} incidenten)",
        zorder=4,
    )

    # Baseline reference
    ax.axhline(
        baseline,
        color=color_base, linewidth=1.6, linestyle="--",
        label=f"Baseline (rustige vensters) = {baseline:.2f}",
    )

    # Incident marker + annotation above the peak
    ax.axvline(0, color=color_line, linewidth=1.8, linestyle=":", alpha=0.85)
    peak_idx_es = int(np.argmax(mean_profile))
    peak_lag = lag_hours[peak_idx_es]
    peak_val_es = float(mean_profile[peak_idx_es])
    ax.annotate(
        f"+{pct_increase:.0f}% t.o.v. baseline",
        xy=(peak_lag, peak_val_es),
        xytext=(peak_lag + 0.40, peak_val_es * 0.80),
        fontsize=9.5, fontweight="semibold",
        color=color_line,
        arrowprops=dict(arrowstyle="->,head_width=0.25", color=color_line, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_line, lw=0.7, alpha=0.9),
    )

    ax.set_xlabel(f"Tijd t.o.v. incident (uren, venster = {window_minutes} min)", fontsize=11)
    ax.set_ylabel("Gem. berichten per venster", fontsize=11)
    ax.set_title(
        "De groep reageert direct en gecoördineerd op een incident",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(lag_hours[::2])
    ax.set_xticklabels([f"{h:+.1f}u" for h in lag_hours[::2]], fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, color=_MATPLOTLIB_GRID_COLOR)
    ax.grid(axis="x", visible=False)

    ax.text(
        0.98, 0.02,
        "Observationeel \u00b7 geen causaliteit \u00b7 gemiddeld profiel over alle incidenten",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=DEFAULT_PLOT_SETTINGS.muted_text_color, style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def _build_incident_pattern() -> str:
    """Build a regex pattern that matches incident bag-of-words terms.

    :return: Regex pattern for incident-term matching.
    :rtype: str
    """
    terms: list[str] = [str(term) for term in INCIDENT_BOW_TERMS]
    escaped_terms: list[str] = [
        re.escape(str(term)).replace(r"\ ", r"\s+") for term in terms
    ]
    return r"\b(?:{})\b".format("|".join(escaped_terms))


def _flag_incident_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Add incident message flags to a dataframe copy.

    :param df: Input dataframe containing ``message``.
    :type df: pd.DataFrame
    :return: Dataframe with ``is_incident_message`` column.
    :rtype: pd.DataFrame
    """
    pattern = _build_incident_pattern()
    working = df.copy()
    working["is_incident_message"] = (
        working["message"]
        .fillna("")
        .astype(str)
        .str.contains(pattern, case=False, regex=True, na=False)
        .astype(int)
    )
    return working


def _prepare_weekly_incident_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly totals and incident counts from message-level data.

    :param df: Input dataframe containing ``datetime`` and ``message``.
    :type df: pd.DataFrame
    :return: Weekly aggregated dataframe with incident metrics.
    :rtype: pd.DataFrame
    """
    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")
    if "message" not in df.columns:
        raise KeyError("message column not found.")

    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])

    if "date_only" not in working.columns:
        working["date_only"] = working["datetime"].map(
            lambda value: value.date() if pd.notna(value) else None
        )
    else:
        working["date_only"] = pd.to_datetime(working["date_only"], errors="coerce")

    working = _flag_incident_messages(working)

    daily = (
        working.groupby("date_only")
        .agg(
            total_message_count=("datetime", "size"),
            incident_message_count=("is_incident_message", "sum"),
        )
        .sort_index()
    )

    weekly = (
        daily.resample("W-MON")
        .sum(numeric_only=True)
        .rename_axis("week_start")
        .sort_index()
    )
    weekly["incident_ratio_pct"] = (
        weekly["incident_message_count"]
        / weekly["total_message_count"].replace(0, pd.NA)
        * 100
    ).fillna(0.0)
    return weekly


def plot_incident_discussion_timeline(
    df: pd.DataFrame,
    out_path: str | Path = "img/incident_discussion_timeline.png",
) -> None:
    """Plot weekly total versus incident message volume with keyword context.

    :param df: Input dataframe with message and datetime features.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    focus = _prepare_weekly_incident_df(df)
    if bool(focus.empty):
        return

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()

    fig, ax = plt.subplots(figsize=(14, 5.6))
    total_color = DEFAULT_PLOT_SETTINGS.neutral_color
    incident_color = DEFAULT_PLOT_SETTINGS.danger_color
    incident_text_color = "#8e0000"

    # --- Background total activity bars ---
    bars_total = ax.bar(
        focus.index,
        focus["total_message_count"],
        color=total_color,
        alpha=0.50,
        width=6.4,
        zorder=1,
        label="Totale berichten / week",
    )

    # --- Incident volume bars (overlaid, narrower) ---
    bars_incident = ax.bar(
        focus.index,
        focus["incident_message_count"],
        color=incident_color,
        alpha=0.85,
        width=4.0,
        zorder=4,
        label="Incidentberichten / week",
    )

    ax.set_ylabel("Berichten / week")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # --- Incident ratio on secondary y-axis ---
    ratio_line = ax.twinx()
    ratio_line.plot(
        focus.index,
        focus["incident_ratio_pct"],
        color=incident_text_color,
        linewidth=1.8,
        alpha=0.80,
        marker="o",
        markersize=3.5,
        label="Incidentratio (%)",
        zorder=6,
    )
    ratio_line.set_ylabel("Incidentratio (%)", color=incident_text_color,
                          fontsize=DEFAULT_PLOT_SETTINGS.axis_label_fontsize)
    ratio_line.tick_params(axis="y", colors=incident_text_color,
                           labelsize=DEFAULT_PLOT_SETTINGS.tick_fontsize)
    ratio_line.set_ylim(bottom=0)
    ratio_line.spines["top"].set_visible(False)

    date_min = pd.to_datetime(focus.index.min(), errors="coerce")
    date_max = pd.to_datetime(focus.index.max(), errors="coerce")
    span_text = (
        f"{date_min.strftime('%d-%m-%Y')} t/m {date_max.strftime('%d-%m-%Y')}"
        if bool(pd.notna(date_min)) and bool(pd.notna(date_max))
        else "gehele periode"
    )
    ax.set_title(
        "Incidentweken vallen samen met hogere groepsactiviteit\n"
        f"{span_text} · grijs=totale chat · rood=incidentberichten · lijn=incidentratio"
    )
    ax.set_xlabel("Week")
    y_max = max(
        float(focus["total_message_count"].max()),
        float(focus["incident_message_count"].max()) * 2.0,
    )
    ax.set_ylim(0, max(25, y_max * 1.12))

    handles = [bars_total, bars_incident, ratio_line.lines[0]]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, frameon=False, loc="upper left",
              fontsize=DEFAULT_PLOT_SETTINGS.legend_fontsize, ncol=3)

    working = _flag_incident_messages(df)
    incident_msgs = (
        working.loc[working["is_incident_message"] == 1, "message"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    term_patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            term,
            re.compile(
                r"\b" + re.escape(str(term)).replace(r"\ ", r"\s+") + r"\b",
                flags=re.IGNORECASE,
            ),
        )
        for term in sorted(INCIDENT_BOW_TERMS, key=lambda item: len(str(item)), reverse=True)
    ]
    term_counts: dict[str, int] = {}
    for msg in incident_msgs:
        for term, pattern in term_patterns:
            if pattern.search(str(msg)):
                term_counts[term] = term_counts.get(term, 0) + 1
                break
    if term_counts:
        top_items = sorted(term_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        top_terms_text = "\n".join(f"{term}: {count}" for term, count in top_items)
        ax.text(
            0.99,
            0.96,
            f"Top flagged words\n{top_terms_text}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d9d9d9", "boxstyle": "round,pad=0.35"},
        )

    if len(focus.index) > 0:
        start = pd.to_datetime(focus.index.min(), errors="coerce")
        end = pd.to_datetime(focus.index.max(), errors="coerce")
        if bool(pd.notna(start)) and bool(pd.notna(end)):
            if bool(start == end):
                tick_dates = [start]
            else:
                tick_dates = list(pd.date_range(start=start, end=end, periods=9))
            ax.set_xticks(mdates.date2num(tick_dates))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.subplots_adjust(bottom=0.10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


def plot_incident_activity_correlation(
    df: pd.DataFrame,
    out_path: str | Path = "img/incident_activity_correlation.png",
) -> None:
    """Plot correlation between weekly total chat activity and incident message count.

    :param df: Input dataframe with message and datetime features.
    :type df: pd.DataFrame
    :param out_path: Output path for the rendered image.
    :type out_path: str | Path
    :return: None.
    :rtype: None
    """
    weekly = _prepare_weekly_incident_df(df)
    if bool(weekly.empty):
        return

    corr = weekly["total_message_count"].corr(weekly["incident_message_count"])
    n_points = len(weekly)
    p_label = "p n.v.t."

    highlight_cutoff = float(weekly["incident_ratio_pct"].quantile(0.9))
    highlighted = weekly["incident_ratio_pct"] >= highlight_cutoff

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()

    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(42)
    y_jitter = rng.normal(loc=0.0, scale=0.08, size=n_points)
    ax.scatter(
        weekly["total_message_count"],
        weekly["incident_message_count"] + y_jitter,
        c=np.where(highlighted, DEFAULT_PLOT_SETTINGS.danger_color, DEFAULT_PLOT_SETTINGS.neutral_color),
        alpha=0.70,
        s=55,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    info_text = f"n = {n_points}\nr = {corr:.2f}\n{p_label}"
    x_values = weekly["total_message_count"].astype(float).to_numpy()
    y_values = weekly["incident_message_count"].astype(float).to_numpy()
    if len(x_values) >= 2 and (x_values.max() - x_values.min()) > 0:
        reg = stats.linregress(x_values, y_values)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        p_val = float(reg.pvalue)
        p_label = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        info_text = f"n = {n_points}\nr = {corr:.2f}\n{p_label}"
        x_line = pd.Series([x_values.min(), x_values.max()])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="#777777",
            linewidth=1.6,
            linestyle="--",
            alpha=0.85,
            zorder=4,
        )
        info_text = f"{info_text}\ny = {slope:.3f}x + {intercept:.2f}"
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.annotation_fontsize,
        color=DEFAULT_PLOT_SETTINGS.text_color,
        bbox=DEFAULT_PLOT_SETTINGS.annotation_box,
    )
    ax.set_title(
        "Incidentberichten vallen samen met hogere activiteit per week\n"
        f"Patroon zichtbaar, zonder causaliteitsclaim (r = {corr:.2f}, {p_label})"
    )
    ax.set_xlabel("Totale berichten / week")
    ax.set_ylabel("Incidentberichten / week")
    ax.set_axisbelow(True)
    ax.text(
        0.02,
        0.98,
        "Rood = weken met relatief hoog incidentaandeel (top 10%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=DEFAULT_PLOT_SETTINGS.caption_fontsize,
        color=DEFAULT_PLOT_SETTINGS.muted_text_color,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)
