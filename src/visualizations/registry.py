"""Registry loader and dispatcher for visualization entry points."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

VisualizationFn = Callable[..., None]

MODULES = [
    "src.visualizations.emoji",
    "src.visualizations.negative_reactions",
    "src.visualizations.chat_activity",
    "src.visualizations.response_time_suite",
    "src.visualizations.incident_timeline",
]


def _load_registry() -> dict[str, VisualizationFn]:
    """Load and merge REGISTRY dictionaries from visualization modules."""
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
    """Run selected visualizations using the module registry."""
    registry = _load_registry()

    for name in selections:
        if name not in registry:
            logger.warning("Visualization not found: %s", name)

    for name, enabled in selections.items():
        if enabled and name in registry:
            logger.info("Visualization: %s", name)
            try:
                registry[name](df, out_dir=out_dir)
            except (KeyError, ValueError, RuntimeError, OSError):
                logger.exception("Visualization failed: %s", name)
