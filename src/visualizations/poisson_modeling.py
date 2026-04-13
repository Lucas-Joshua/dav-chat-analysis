"""Public wrappers and registry for Poisson statistical modeling (Les 5)."""

from __future__ import annotations

from pathlib import Path

from src.visualizations._poisson_plotters import plot_poisson_model
from src.visualizations.utils import resolve_lesson_output_path


def poisson_model(df, out_dir: str | Path | None = None) -> None:
    """Generate the Poisson observed-vs-expected distribution and anomaly plot.

    :param df: Processed chat dataframe.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_poisson_model(
        df,
        out_path=resolve_lesson_output_path(
            out_dir,
            "poisson_model",
            "poisson_model.png",
        ),
    )


REGISTRY = {
    "poisson_model": poisson_model,
}
