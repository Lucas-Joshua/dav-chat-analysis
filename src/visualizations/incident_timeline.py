from __future__ import annotations

from pathlib import Path

from src.visualizations._incident_plotters import (
    plot_incident_activity_correlation,
    plot_incident_discussion_timeline,
)
from src.visualizations.utils import resolve_output_path


def incident_discussion_timeline(df, out_dir: str | Path | None = None) -> None:
    """Generate incident/safety discussion timeline chart."""
    plot_incident_discussion_timeline(
        df,
        out_path=resolve_output_path(out_dir, "incident_discussion_timeline.png"),
    )


def incident_activity_correlation(df, out_dir: str | Path | None = None) -> None:
    """Generate weekly activity vs incident correlation chart."""
    plot_incident_activity_correlation(
        df,
        out_path=resolve_output_path(out_dir, "incident_activity_correlation.png"),
    )


REGISTRY = {
    "incident_discussion_timeline": incident_discussion_timeline,
    "incident_activity_correlation": incident_activity_correlation,
}
