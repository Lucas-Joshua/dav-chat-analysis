from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re

from src.modules.feature_engineering import INCIDENT_BOW_TERMS


def _prepare_weekly_incident_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly totals and incident counts from message-level data using BOW flags."""
    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")
    if "message" not in df.columns:
        raise KeyError("message column not found.")

    working = df.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"])

    if "date_only" not in working.columns:
        working["date_only"] = working["datetime"].dt.floor("D")
    else:
        working["date_only"] = pd.to_datetime(working["date_only"], errors="coerce")

    bow_pattern = r"\b(?:{})\b".format(
        "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in INCIDENT_BOW_TERMS)
    )
    working["is_incident_message"] = (
        working["message"]
        .fillna("")
        .astype(str)
        .str.contains(bow_pattern, case=False, regex=True, na=False)
        .astype(int)
    )

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
    """Plot weekly total vs incident message volume with BOW keyword context."""
    focus = _prepare_weekly_incident_df(df)
    if focus.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    total_color = "#b0b0b0"
    incident_color = "#c62828"
    incident_text_color = "#8e0000"

    bars_total = ax.bar(
        focus.index,
        focus["total_message_count"],
        color=total_color,
        alpha=0.55,
        width=6.4,
        zorder=1,
        label="Total messages / week",
    )
    incident_week_mask = focus["incident_message_count"] > 0
    if incident_week_mask.any():
        ax.bar(
            focus.index[incident_week_mask],
            focus.loc[incident_week_mask, "total_message_count"],
            width=6.4,
            facecolor="none",
            edgecolor="#3f444a",
            linewidth=0.6,
            alpha=0.7,
            zorder=2,
        )
    ax.set_ylabel("Messages / week")
    ax.grid(axis="y", alpha=0.14, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bars_incident = ax.bar(
        focus.index,
        focus["incident_message_count"],
        color=incident_color,
        width=3.8,
        zorder=4,
        label="Incident messages / week",
    )
    for x, cnt in zip(focus.index, focus["incident_message_count"]):
        if cnt > 0:
            ax.text(
                x,
                cnt + 0.8,
                f"{int(cnt)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=incident_text_color,
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "boxstyle": "round,pad=0.15",
                },
            )

    ax.set_title("When Incidents Happen, Chat Activity Increases")
    ax.set_xlabel("Week")
    ax.set_ylim(0, 175)

    handles = [bars_total, bars_incident]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=9)

    # Show top flagged words as a vertical list on the right side.
    working = df.copy()
    bow_pattern = r"\b(?:{})\b".format(
        "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in INCIDENT_BOW_TERMS)
    )
    working["is_incident_message"] = (
        working["message"]
        .fillna("")
        .astype(str)
        .str.contains(bow_pattern, case=False, regex=True, na=False)
        .astype(int)
    )
    incident_msgs = working.loc[working["is_incident_message"] == 1, "message"].fillna("").astype(str)
    term_patterns = [
        (
            term,
            re.compile(
                r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b",
                flags=re.IGNORECASE,
            ),
        )
        for term in sorted(INCIDENT_BOW_TERMS, key=len, reverse=True)
    ]
    term_counts: dict[str, int] = {}
    for msg in incident_msgs:
        for term, pattern in term_patterns:
            if pattern.search(msg):
                term_counts[term] = term_counts.get(term, 0) + 1
                break
    if term_counts:
        top_items = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
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
        start = focus.index.min()
        end = focus.index.max()
        if start == end:
            ticks = [start]
        else:
            ticks = pd.date_range(start=start, end=end, periods=9)
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.subplots_adjust(bottom=0.10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_incident_activity_correlation(
    df: pd.DataFrame,
    out_path: str | Path = "img/incident_activity_correlation.png",
) -> None:
    """Plot correlation between weekly total chat activity and incident message count."""
    weekly = _prepare_weekly_incident_df(df)
    if weekly.empty:
        return

    corr = weekly["total_message_count"].corr(weekly["incident_message_count"])
    n_points = len(weekly)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        weekly["total_message_count"],
        weekly["incident_message_count"],
        c=weekly["incident_ratio_pct"],
        cmap="Reds",
        alpha=0.85,
        s=38,
        edgecolors="white",
        linewidths=0.4,
    )
    info_text = f"n = {n_points}\ncorr = {corr:.2f}"
    x = weekly["total_message_count"].astype(float).to_numpy()
    y = weekly["incident_message_count"].astype(float).to_numpy()
    if len(x) >= 2 and (x.max() - x.min()) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = pd.Series([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="#1f1f1f",
            linewidth=2.0,
            alpha=0.9,
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
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )
    ax.set_title("Weekly Activity vs Incident Count")
    ax.set_xlabel("Total messages / week")
    ax.set_ylabel("Incident messages / week")
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
