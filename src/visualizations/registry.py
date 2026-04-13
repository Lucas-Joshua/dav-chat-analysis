"""Registry loader and dispatcher for visualization entry points."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

VisualizationFn = Callable[..., None]

MODULES = [
    "src.visualizations.emoji",
    "src.visualizations.negative_reactions",
    "src.visualizations.chat_activity",
    "src.visualizations.incident_timeline",
    "src.visualizations.incident_context_modeling",
    "src.visualizations.author_clustering",
    "src.visualizations.time_series_modeling",
    "src.visualizations.poisson_modeling",
]


def _load_registry() -> dict[str, VisualizationFn]:
    """Load and merge ``REGISTRY`` dictionaries from visualization modules.

    :return: Mapping from visualization names to callables.
    :rtype: dict[str, VisualizationFn]
    """
    registry: dict[str, VisualizationFn] = {}

    for module_path in MODULES:
        module = importlib.import_module(module_path)
        module_registry = getattr(module, "REGISTRY", {})
        registry.update(module_registry)

    return registry


def run_selected(
    df,
    selections: dict[str, bool],
    out_dir: str | Path | None = None,
) -> None:
    """Run selected visualizations using the module registry.

    :param df: Processed dataframe used by visualization functions.
    :type df: Any
    :param selections: Mapping from visualization name to enabled flag.
    :type selections: dict[str, bool]
    :param out_dir: Optional output directory for generated figures.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    registry = _load_registry()

    for name in selections:
        if name not in registry:
            logger.warning("Visualization not found: %s", name)

    for name, enabled in selections.items():
        if enabled and name in registry:
            logger.info("Visualization: %s", name)
            try:
                registry[name](df, out_dir=out_dir)
            except Exception:
                logger.exception("Visualization failed: %s", name)
