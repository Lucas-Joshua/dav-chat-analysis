"""Preprocessing orchestration for loading, cleaning, and anonymizing chats."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.modules import anonymizer, data_cleaning, data_loader


class ChatPreprocessor:
    """Run the chat preprocessing flow from raw export to anonymized dataframe."""

    def __init__(self) -> None:
        """Initialize the preprocessor with module logger."""
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load(raw_path: str | Path) -> pd.DataFrame:
        """Load raw chat lines from disk."""
        return data_loader.load_raw_chat(Path(raw_path))

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw chat lines into a structured dataframe."""
        return data_cleaning.clean_data(df)

    @staticmethod
    def anonymize(df: pd.DataFrame) -> pd.DataFrame:
        """Replace sender names with pseudonyms."""
        return anonymizer.apply_anonymization(df)

    def run(self, raw_path: str | Path) -> pd.DataFrame:
        """Execute load, clean, and anonymize preprocessing steps."""
        self.logger.info("Step 1/6: load")
        df = self.load(raw_path)

        self.logger.info("Step 2/6: clean")
        df = self.clean(df)

        self.logger.info("Step 3/6: anonymize")
        return self.anonymize(df)
