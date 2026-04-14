"""Incident relationship visualizations and registry entries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy import stats

from src.modules.feature_engineering import INCIDENT_BOW_TERMS
from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS
from src.visualizations.utils import resolve_lesson_output_path


def _build_incident_pattern() -> str:
    """Build a regex pattern that matches incident bag-of-words terms."""
    terms: list[str] = [str(term) for term in INCIDENT_BOW_TERMS]
    escaped_terms: list[str] = [re.escape(str(term)).replace(r"\ ", r"\s+") for term in terms]
    return r"\b(?:{})\b".format("|".join(escaped_terms))


def _flag_incident_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Add incident message flags to a dataframe copy."""
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
    """Build weekly totals and incident counts from message-level data."""
    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")
    if "message" not in df.columns:
        raise KeyError("message column not found.")

    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])

    if "date_only" not in working.columns:
        working["date_only"] = working["datetime"].map(lambda value: value.date() if pd.notna(value) else None)
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
    """Plot weekly total versus incident message volume with keyword context."""
    focus = _prepare_weekly_incident_df(df)
    if bool(focus.empty):
        return

    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)
    DEFAULT_PLOT_SETTINGS.apply_matplotlib_rcparams()

    fig, ax = plt.subplots(figsize=(14, 5.6))
    total_color = DEFAULT_PLOT_SETTINGS.neutral_color
    incident_color = DEFAULT_PLOT_SETTINGS.danger_color
    incident_text_color = "#8e0000"

    bars_total = ax.bar(
        focus.index,
        focus["total_message_count"],
        color=total_color,
        alpha=0.50,
        width=6.4,
        zorder=1,
        label="Totale berichten / week",
    )
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
    ratio_line.set_ylabel("Incidentratio (%)", color=incident_text_color, fontsize=DEFAULT_PLOT_SETTINGS.axis_label_fontsize)
    ratio_line.tick_params(axis="y", colors=incident_text_color, labelsize=DEFAULT_PLOT_SETTINGS.tick_fontsize)
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
    y_max = max(float(focus["total_message_count"].max()), float(focus["incident_message_count"].max()) * 2.0)
    ax.set_ylim(0, max(25, y_max * 1.12))

    handles = [bars_total, bars_incident, ratio_line.lines[0]]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=DEFAULT_PLOT_SETTINGS.legend_fontsize, ncol=3)

    working = _flag_incident_messages(df)
    incident_msgs = working.loc[working["is_incident_message"] == 1, "message"].fillna("").astype(str).tolist()
    term_patterns = [
        (
            term,
            re.compile(r"\b" + re.escape(str(term)).replace(r"\ ", r"\s+") + r"\b", flags=re.IGNORECASE),
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
            tick_dates = [start] if bool(start == end) else list(pd.date_range(start=start, end=end, periods=9))
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
    """Plot correlation between weekly total chat activity and incident message count."""
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
        ax.plot(x_line, y_line, color="#777777", linewidth=1.6, linestyle="--", alpha=0.85, zorder=4)
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


def incident_discussion_timeline(df, out_dir: str | Path | None = None) -> None:
    """Generate the incident and safety discussion timeline chart."""
    plot_incident_discussion_timeline(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "incident_discussion_timeline",
            "incident_discussion_timeline.png",
        ),
    )


def incident_activity_correlation(df, out_dir: str | Path | None = None) -> None:
    """Generate the weekly activity versus incident correlation chart."""
    plot_incident_activity_correlation(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "incident_activity_correlation",
            "incident_activity_correlation.png",
        ),
    )


REGISTRY = {
    "incident_discussion_timeline": incident_discussion_timeline,
    "incident_activity_correlation": incident_activity_correlation,
}
