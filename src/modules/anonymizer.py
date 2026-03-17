"""Utilities for generating and applying pseudonymous user mappings."""

import pandas as pd
import logging
from random import Random
from src import config

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_name_lists() -> tuple[list[str], list[str]]:
    """Load first and last name lists from local assets."""
    first_names = pd.read_csv(config.FIRST_NAMES_FILE).iloc[:, 0].dropna().tolist()
    last_names = pd.read_csv(config.LAST_NAMES_FILE).iloc[:, 0].dropna().tolist()

    if not first_names or not last_names:
        raise ValueError("Name files contain no valid data")

    return first_names, last_names


def generate_fake_names(n: int) -> list[str]:
    """Generate a list of unique fake full names."""
    rng = Random(RANDOM_SEED)

    first_names, last_names = load_name_lists()

    if n > min(len(first_names), len(last_names)):
        raise ValueError("Not enough names available for anonymization")

    rng.shuffle(first_names)
    rng.shuffle(last_names)

    return [f"{first_names[i]} {last_names[i]}" for i in range(n)]


def load_user_mapping() -> pd.DataFrame | None:
    """Load an existing user mapping file if available."""
    if not config.USER_MAPPING_FILE.exists():
        return None
    return pd.read_csv(config.USER_MAPPING_FILE)


def save_user_mapping(mapping_df: pd.DataFrame) -> None:
    """Save the user mapping dataframe to disk."""
    mapping_df.to_csv(config.USER_MAPPING_FILE, index=False)
    logger.info(f"User mapping saved to {config.USER_MAPPING_FILE}")


def create_mapping(users: list[str]) -> pd.DataFrame:
    """Create a mapping dataframe from real users to fake names."""
    fake_names = generate_fake_names(len(users))

    return pd.DataFrame(
        {
            "real_name": users,
            "pseudo_name": fake_names,
        }
    )


def apply_anonymization(df: pd.DataFrame) -> pd.DataFrame:
    """Replace real sender names with pseudonyms."""

    if "sender" not in df.columns:
        raise ValueError("Column 'sender' not found")

    df_copy = df.copy()

    users = sorted(df_copy["sender"].dropna().unique())

    mapping_df = load_user_mapping()

    if mapping_df is None:
        logger.info("Creating new anonymization mapping")
        mapping_df = create_mapping(users)
    else:
        existing_users = set(mapping_df["real_name"])
        new_users = [u for u in users if u not in existing_users]

        if new_users:
            logger.info(f"New users detected: {new_users}")
            new_mapping = create_mapping(new_users)
            mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)

    save_user_mapping(mapping_df)

    mapping_dict = dict(zip(mapping_df["real_name"], mapping_df["pseudo_name"]))

    df_copy["sender"] = df_copy["sender"].map(mapping_dict)

    if df_copy["sender"].isna().any():
        raise ValueError("Anonymization failed due to missing mappings")

    return df_copy
