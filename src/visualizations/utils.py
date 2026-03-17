"""Shared utility helpers for visualization output management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def resolve_output_path(out_dir: str | Path | None, filename: str) -> str | Path:
    """Build a plot output path using the configured output directory."""
    return Path(out_dir) / filename if out_dir else f"img/{filename}"


def ensure_parent_dir(out_path: str | Path) -> Path:
    """Ensure the output directory exists and return the resolved path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_user_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Resolve the user column name from the dataframe."""
    if preferred and preferred in df.columns:
        return preferred
    if "user" in df.columns:
        return "user"
    if "sender" in df.columns:
        return "sender"
    raise KeyError("No user column found.")


def top_user_order(counts: pd.DataFrame, user_col: str, top_users: int) -> list[str]:
    """Return top users ordered by descending total count."""
    totals = counts.groupby(user_col, as_index=True)["count"].sum()
    return totals.sort_values(ascending=False).head(top_users).index.tolist()
