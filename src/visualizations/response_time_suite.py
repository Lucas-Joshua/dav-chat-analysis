"""Response-time visualization suite generation and registry entry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualizations.plot_settings import DEFAULT_PLOT_SETTINGS

PLOT_NAME = "response_time_suite"


def _compute_response_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute response-delay metrics for valid sender switches.

    :param df: Input dataframe with at least ``datetime`` and ``sender``.
    :type df: pd.DataFrame
    :return: Dataframe with response-time and derived temporal features.
    :rtype: pd.DataFrame
    """
    df = df.copy()

    if "datetime" not in df.columns:
        raise KeyError("datetime column not found.")
    if "sender" not in df.columns:
        raise KeyError("sender column not found.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    run_id = (df["sender"] != df["sender"].shift()).cumsum()
    run_start = df.groupby(run_id)["datetime"].first()
    next_run_start = run_start.shift(-1)

    df["next_other_sender_time"] = run_id.map(next_run_start)
    df["response_time"] = (
        df["next_other_sender_time"] - df["datetime"]
    ).dt.total_seconds()

    df = df.loc[
        (df["response_time"] > 30) & (df["response_time"] < 3600)
    ].copy()

    df["response_minutes"] = df["response_time"] / 60.0

    datetime_series = pd.to_datetime(df["datetime"], errors="coerce")
    if "hour" not in df.columns:
        df["hour"] = datetime_series.map(
            lambda value: int(value.hour) if pd.notna(value) else None
        )
    if "day_of_week" not in df.columns:
        df["day_of_week"] = datetime_series.map(
            lambda value: value.strftime("%A") if pd.notna(value) else None
        )
    if "message_length" not in df.columns:
        msg_col = "message" if "message" in df.columns else "original_message"
        if msg_col in df.columns:
            df["message_length"] = df[msg_col].astype(str).str.len()
    if "date_only" not in df.columns:
        df["date_only"] = datetime_series.map(
            lambda value: value.date() if pd.notna(value) else None
        )

    return df


def generate(df: pd.DataFrame, out_dir: str | Path | None = None) -> None:
    """Generate the full response-time visualization suite.

    :param df: Processed chat dataframe.
    :type df: pd.DataFrame
    :param out_dir: Optional output directory root.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    import matplotlib.pyplot as plt

    df = _compute_response_minutes(df)

    output_root = Path(out_dir) if out_dir else Path("img")
    output_dir = output_root / "response_time"
    output_dir.mkdir(parents=True, exist_ok=True)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    hours = list(range(24))
    plt.style.use(DEFAULT_PLOT_SETTINGS.matplotlib_style)

    median_by_day_hour = (
        df.groupby(["day_of_week", "hour"])["response_minutes"]
        .median()
        .unstack("hour")
        .reindex(index=day_order, columns=hours)
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    for idx, day in enumerate(day_order):
        ax = axes[idx]
        day_series = median_by_day_hour.loc[day]
        valid = day_series.dropna()
        if not valid.empty:
            start_hour = int(valid.index.min())
            end_hour = int(valid.index.max())
            active_range = list(range(start_hour, end_hour + 1))
            ax.plot(
                active_range,
                day_series.loc[active_range],
                color=DEFAULT_PLOT_SETTINGS.primary_color,
                linewidth=2,
            )
            ax.scatter(
                valid.index,
                valid.values,
                color=DEFAULT_PLOT_SETTINGS.primary_color,
                s=18,
                alpha=0.8,
            )
        ax.set_title(f"Daily Chat Response Pattern — {day}")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Median response (minutes)")

    axes[-1].axis("off")

    fig.savefig(output_dir / "daily_response_pattern_small_multiples.png", dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_summary = (
        df.groupby("hour")["response_minutes"]
        .agg(median_response="median", response_count="size")
        .reindex(hours)
    )
    fast_threshold = hourly_summary["median_response"].quantile(0.15)
    fast_mask = hourly_summary["median_response"] <= fast_threshold

    size_scale = hourly_summary["response_count"].fillna(0) * 0.8
    marker_sizes = size_scale.clip(lower=30)

    regular = hourly_summary[~fast_mask].dropna(subset=["median_response"])
    fast = hourly_summary[fast_mask].dropna(subset=["median_response"])

    ax.scatter(
        regular.index,
        regular["median_response"],
        s=marker_sizes.loc[regular.index],
        color=DEFAULT_PLOT_SETTINGS.neutral_color,
        alpha=0.7,
        linewidths=0,
        zorder=1,
        label="Other hours",
    )
    ax.scatter(
        fast.index,
        fast["median_response"],
        s=marker_sizes.loc[fast.index],
        color=DEFAULT_PLOT_SETTINGS.success_color,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
        label="Fast hours (bottom 15%)",
    )

    ax.set_title("Fast and Active Chat Hours")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Median response (minutes)")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 1))
    ax.grid(True, which="major", axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    fig.savefig(output_dir / "overlay_response_trend.png", dpi=DEFAULT_PLOT_SETTINGS.dpi)
    fig.savefig(output_dir / "response_time_spike_hours_line.png", dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["response_minutes"], bins=30, color=DEFAULT_PLOT_SETTINGS.accent_color, edgecolor="white")
    ax.set_title("Response Time Distribution")
    ax.set_xlabel("Response time (minutes)")
    ax.set_ylabel("Count")

    fig.savefig(output_dir / "response_time_distribution.png", dpi=DEFAULT_PLOT_SETTINGS.dpi)
    plt.close(fig)


REGISTRY = {
    "response_time_suite": generate,
}
