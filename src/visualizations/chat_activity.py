"""Public chat-activity visualization wrappers and registry entries."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._activity_plotters import (
    plot_chat_activity_by_hour,
    plot_chat_activity_distribution,
)
from src.visualizations.utils import resolve_output_path


def chat_activity_by_hour(df, out_dir: str | Path | None = None) -> None:
    """Generate the chat activity by hour line chart."""
    plot_chat_activity_by_hour(
        df,
        out_path=resolve_output_path(out_dir, "chat_activity_by_hour.png"),
    )


def chat_activity_distribution(df, out_dir: str | Path | None = None) -> None:
    """Generate the chat activity distribution bar chart."""
    plot_chat_activity_distribution(
        df,
        output=Path(resolve_output_path(out_dir, "plot_chat_activity_distribution.png")),
    )


REGISTRY = {
    "chat_activity_by_hour": chat_activity_by_hour,
    "chat_activity_distribution": chat_activity_distribution,
}
