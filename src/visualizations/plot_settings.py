"""Centralized base settings for plot styling.

Gestalt principles applied throughout:
- Proximity:    consistent spacing between title, axes, and annotations
- Similarity:   unified color palette, font sizes, and gridline style
- Figure/Ground: white backgrounds with light gridlines, data always foreground
- Common fate:  same color = same meaning across all charts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlotSettings:
    """Shared baseline settings for Plotly and Matplotlib charts.

    :ivar plot_bgcolor: Plot background color.
    :vartype plot_bgcolor: str
    :ivar paper_bgcolor: Figure background color.
    :vartype paper_bgcolor: str
    :ivar gridcolor: Gridline color.
    :vartype gridcolor: str
    :ivar font_size: Default Plotly font size.
    :vartype font_size: int
    """

    # ── Backgrounds ───────────────────────────────────────────────────────────
    plot_bgcolor: str = "white"
    paper_bgcolor: str = "white"
    gridcolor: str = "rgba(0,0,0,0.07)"

    # ── Font sizes (Gestalt: Similarity — same hierarchy everywhere) ──────────
    font_size: int = 13          # Plotly default (tick labels, legend)
    title_fontsize: int = 13     # Matplotlib ax.set_title
    axis_label_fontsize: int = 11
    tick_fontsize: int = 10
    annotation_fontsize: int = 10
    caption_fontsize: int = 9
    legend_fontsize: int = 10

    # ── Output quality ────────────────────────────────────────────────────────
    dpi: int = 180

    # ── Plotly / Matplotlib base templates ───────────────────────────────────
    plotly_template: str = "simple_white"
    matplotlib_style: str = "seaborn-v0_8-white"   # white bg, no default grid

    # ── Layout ────────────────────────────────────────────────────────────────
    margin: dict[str, int] = field(
        default_factory=lambda: {"l": 65, "r": 50, "t": 85, "b": 65}
    )

    # ── Color palette (Common fate: color = meaning) ─────────────────────────
    neutral_color: str = "#ABABAB"     # background / baseline data
    primary_color: str = "#2F6DB3"     # trend lines, secondary emphasis
    success_color: str = "#2E7D32"
    danger_color: str = "#C62828"      # highlights, incidents, peaks
    accent_color: str = "#6C4BAF"
    text_color: str = "#1A1A1A"
    muted_text_color: str = "#555555"

    # ── Emoji category colors ─────────────────────────────────────────────────
    emoji_group_colors: dict[str, str] = field(default_factory=lambda: {
        "humor": "#E8A800",
        "positive": "#2E9E52",
        "social": "#3A7FC1",
        "negative_reflective": "#C62828",
    })

    # ── Annotation box style (Gestalt: Enclosure) ────────────────────────────
    @property
    def annotation_box(self) -> dict[str, Any]:
        """Consistent annotation bounding-box kwargs for Matplotlib."""
        return dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.8, alpha=0.95)

    def apply_matplotlib_rcparams(self) -> None:
        """Push all style constants into Matplotlib's global rcParams.

        Call this once per figure, after ``plt.style.use()``, so our values
        take precedence over the base stylesheet.

        :return: None.
        :rtype: None
        """
        import matplotlib as mpl
        mpl.rcParams.update({
            # Backgrounds
            "axes.facecolor": self.plot_bgcolor,
            "figure.facecolor": self.paper_bgcolor,
            # Typography
            "font.family": "sans-serif",
            "axes.titlesize": self.title_fontsize,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": self.axis_label_fontsize,
            "xtick.labelsize": self.tick_fontsize,
            "ytick.labelsize": self.tick_fontsize,
            "legend.fontsize": self.legend_fontsize,
            # Grid — y-axis only, very light (Figure/Ground)
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            # Spines — remove top & right (less ink, more data)
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#AAAAAA",
            "axes.linewidth": 0.8,
        })

    def base_plotly_layout(self, **overrides: Any) -> dict[str, Any]:
        """Return a consistent base Plotly layout dictionary.

        Includes ``template`` so every Plotly chart inherits ``simple_white``
        without needing an extra ``update_layout(template=...)`` call.

        :param overrides: Layout key-value overrides.
        :type overrides: Any
        :return: Base layout dictionary merged with overrides.
        :rtype: dict[str, Any]
        """
        layout: dict[str, Any] = {
            "template": self.plotly_template,
            "font": {"size": self.font_size, "color": self.text_color},
            "plot_bgcolor": self.plot_bgcolor,
            "paper_bgcolor": self.paper_bgcolor,
            "margin": dict(self.margin),
        }
        layout.update(overrides)
        return layout


DEFAULT_PLOT_SETTINGS = PlotSettings()
