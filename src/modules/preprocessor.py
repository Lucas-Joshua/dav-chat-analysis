"""Preprocessing orchestration for loading, cleaning, and anonymizing chats."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.modules import anonymizer, data_cleaning, data_loader


class ChatPreprocessor:
    """Run the chat preprocessing flow from raw export to anonymized dataframe.

    :ivar logger: Module logger used for preprocessing progress.
    :vartype logger: logging.Logger
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with a module logger.

        :return: None.
        :rtype: None
        """
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load(raw_path: str | Path) -> pd.DataFrame:
        """Load raw chat lines from disk.

        :param raw_path: Path to the raw chat export file.
        :type raw_path: str | Path
        :return: Raw dataframe with one line per row.
        :rtype: pd.DataFrame
        """
        return data_loader.load_raw_chat(Path(raw_path))

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw chat lines into a structured dataframe.

        :param df: Raw dataframe from ``load``.
        :type df: pd.DataFrame
        :return: Structured cleaned dataframe.
        :rtype: pd.DataFrame
        """
        return data_cleaning.clean_data(df)

    @staticmethod
    def anonymize(df: pd.DataFrame) -> pd.DataFrame:
        """Replace sender names with pseudonyms.

        :param df: Cleaned dataframe containing sender names.
        :type df: pd.DataFrame
        :return: Anonymized dataframe.
        :rtype: pd.DataFrame
        """
        return anonymizer.apply_anonymization(df)

    def run(self, raw_path: str | Path) -> pd.DataFrame:
        """Execute load, clean, and anonymize preprocessing steps.

        :param raw_path: Path to the raw chat export file.
        :type raw_path: str | Path
        :return: Fully preprocessed dataframe.
        :rtype: pd.DataFrame
        """
        self.logger.info("Step 1/6: load")
        df = self.load(raw_path)

        self.logger.info("Step 2/6: clean")
        df = self.clean(df)

        self.logger.info("Step 3/6: anonymize")
        return self.anonymize(df)
