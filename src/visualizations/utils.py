from __future__ import annotations

from pathlib import Path


def resolve_output_path(out_dir: str | Path | None, filename: str) -> str | Path:
    """Build a plot output path using the configured output directory."""
    return Path(out_dir) / filename if out_dir else f"img/{filename}"
