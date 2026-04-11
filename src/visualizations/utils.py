"""Shared utility helpers for visualization output management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def resolve_output_path(out_dir: str | Path | None, filename: str) -> str | Path:
    """Build a plot output path using the configured output directory.

    :param out_dir: Optional output directory root.
    :type out_dir: str | Path | None
    :param filename: Plot filename.
    :type filename: str
    :return: Resolved output path.
    :rtype: str | Path
    """
    return Path(out_dir) / filename if out_dir else f"img/{filename}"


def ensure_parent_dir(out_path: str | Path) -> Path:
    """Ensure the output directory exists and return the resolved path.

    :param out_path: Target file path.
    :type out_path: str | Path
    :return: Path object with ensured parent directory.
    :rtype: Path
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_user_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Resolve the user column name from the dataframe.

    :param df: Input dataframe with either ``user`` or ``sender`` column.
    :type df: pd.DataFrame
    :param preferred: Preferred column name when present.
    :type preferred: Optional[str]
    :return: Chosen user column name.
    :rtype: str
    """
    if preferred and preferred in df.columns:
        return preferred
    if "user" in df.columns:
        return "user"
    if "sender" in df.columns:
        return "sender"
    raise KeyError("No user column found.")


def top_user_order(counts: pd.DataFrame, user_col: str, top_users: int) -> list[str]:
    """Return top users ordered by descending total count.

    :param counts: Dataframe containing per-user counts.
    :type counts: pd.DataFrame
    :param user_col: Name of the user column in ``counts``.
    :type user_col: str
    :param top_users: Number of users to keep.
    :type top_users: int
    :return: Ordered list of top user identifiers.
    :rtype: list[str]
    """
    totals = counts.groupby(user_col, as_index=True)["count"].sum()
    return totals.sort_values(ascending=False).head(top_users).index.tolist()
