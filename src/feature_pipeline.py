"""Feature dependency resolution and execution for the chat pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from src.modules.feature_engineering import (
    add_emoji_category,
    add_emoji_features,
    add_incident_bow_features,
    add_time_features,
)

FeatureTransform = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one feature step and the steps it depends on."""

    name: str
    transform: FeatureTransform | None
    depends_on: frozenset[str] = frozenset()


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(name="emoji_features", transform=add_emoji_features),
    FeatureSpec(
        name="emoji_category",
        transform=add_emoji_category,
        depends_on=frozenset({"emoji_features"}),
    ),
    FeatureSpec(name="time_features", transform=add_time_features),
    FeatureSpec(name="incident_bow_features", transform=add_incident_bow_features),
)


FEATURE_EXECUTION_ORDER: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)


class FeaturePipeline:
    """Resolve and apply feature steps in dependency-safe order."""

    def __init__(self, specs: Iterable[FeatureSpec] = FEATURE_SPECS) -> None:
        self.specs = tuple(specs)
        self._spec_by_name = {spec.name: spec for spec in self.specs}

    def resolve(self, requested: Iterable[str]) -> list[str]:
        """Resolve requested features together with their dependencies."""
        resolved: set[str] = set()
        stack = list(requested)

        while stack:
            feature_name = stack.pop()
            if feature_name in resolved:
                continue
            spec = self._spec_by_name.get(feature_name)
            if spec is None:
                raise KeyError(f"Unknown feature operation: {feature_name}")
            resolved.add(feature_name)
            stack.extend(spec.depends_on)

        return [name for name in FEATURE_EXECUTION_ORDER if name in resolved]

    def apply(self, df: pd.DataFrame, operations: Iterable[str]) -> pd.DataFrame:
        """Apply already resolved feature steps to the dataframe."""
        for feature_name in operations:
            transform = self._spec_by_name[feature_name].transform
            if transform is None:
                raise KeyError(f"No transform configured for feature operation: {feature_name}")
            df = transform(df)
        return df
