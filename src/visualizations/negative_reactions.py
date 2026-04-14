"""Public negative-reaction visualization wrappers and registry entries."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._negative_plotters import (
    plot_negative_reaction_concentration,
    plot_negative_reaction_diagnostic,
    plot_negative_reaction_scatter,
)
from src.visualizations.utils import resolve_lesson_output_path


def negative_reaction_concentration(df, out_dir: str | Path | None = None) -> None:
    """Generate the negative-reaction concentration chart.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_negative_reaction_concentration(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "negative_reaction_concentration",
            "negative_reaction_concentration.png",
        ),
    )


def negative_reaction_diagnostic(df, out_dir: str | Path | None = None) -> None:
    """Generate the negative-reaction diagnostic chart.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_negative_reaction_diagnostic(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "negative_reaction_diagnostic",
            "negative_reaction_diagnostic.png",
        ),
    )


def negative_reaction_scatter(df, out_dir: str | Path | None = None) -> None:
    """Generate the negative-reaction scatter chart.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_negative_reaction_scatter(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "negative_reaction_scatter",
            "negative_reaction_scatter.png",
        ),
    )


REGISTRY = {
    "negative_reaction_concentration": negative_reaction_concentration,
    "negative_reaction_diagnostic": negative_reaction_diagnostic,
    "negative_reaction_scatter": negative_reaction_scatter,
}
