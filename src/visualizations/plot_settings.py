"""Centralized base settings for plot styling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlotSettings:
    """Shared baseline settings for Plotly and Matplotlib charts."""

    plot_bgcolor: str = "white"
    paper_bgcolor: str = "white"
    gridcolor: str = "rgba(0,0,0,0.08)"
    font_size: int = 14
    dpi: int = 150
    plotly_template: str = "simple_white"
    matplotlib_style: str = "seaborn-v0_8-whitegrid"
    margin: dict[str, int] = field(default_factory=lambda: {"l": 60, "r": 60, "t": 80, "b": 60})
    neutral_color: str = "#B0B0B0"
    primary_color: str = "#2F6DB3"
    success_color: str = "#2E7D32"
    danger_color: str = "#C62828"
    accent_color: str = "#6C4BAF"
    text_color: str = "#222222"
    muted_text_color: str = "#4B4B4B"
    emoji_group_colors: dict[str, str] = field(default_factory=lambda: {
        "humor": "#F4B400",
        "positive": "#34A853",
        "social": "#4285F4",
        "negative_reflective": "#DB4437",
    })

    def base_plotly_layout(self, **overrides: Any) -> dict[str, Any]:
        """Return a consistent base Plotly layout dictionary."""
        layout: dict[str, Any] = {
            "font": {"size": self.font_size},
            "plot_bgcolor": self.plot_bgcolor,
            "paper_bgcolor": self.paper_bgcolor,
            "margin": dict(self.margin),
        }
        layout.update(overrides)
        return layout


DEFAULT_PLOT_SETTINGS = PlotSettings()
