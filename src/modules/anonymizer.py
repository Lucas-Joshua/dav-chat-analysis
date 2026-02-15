import pandas as pd
import logging
from random import Random
from src import config

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_name_lists() -> tuple[list[str], list[str]]:
    """Load first and last names from CSV files."""
    first_names = pd.read_csv(config.FIRST_NAMES_FILE).iloc[:, 0].dropna().tolist()
    last_names = pd.read_csv(config.LAST_NAMES_FILE).iloc[:, 0].dropna().tolist()

    if not first_names or not last_names:
        raise ValueError("Name files contain no valid data")

    return first_names, last_names

def generate_fake_names(n: int) -> list[str]:
    """Generate n unique fake names."""
    rng = Random(RANDOM_SEED)

    first_names, last_names = load_name_lists()

    if n > len(first_names) or n > len(last_names):
        raise ValueError("Not enough names available for unique combinations")

    rng.shuffle(first_names)
    rng.shuffle(last_names)

    return [f"{first_names[i]} {last_names[i]}" for i in range(n)]

def generate_user_mapping(users: list[str]) -> pd.DataFrame:
    """Create mapping between real users and pseudo names."""
    fake_names = generate_fake_names(len(users))

    return pd.DataFrame({
        "real_name": users,
        "pseudo_name": fake_names
    })

def save_user_mapping(mapping_df: pd.DataFrame) -> None:
    mapping_df.to_csv(config.USER_MAPPING_FILE, index=False)
    logger.info(f"User mapping saved: {config.USER_MAPPING_FILE}")

def load_user_mapping() -> pd.DataFrame | None:
    if not config.USER_MAPPING_FILE.exists():
        return None
    return pd.read_csv(config.USER_MAPPING_FILE)

def apply_anonymization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply anonymization to the 'sender' column."""

    if "sender" not in df.columns:
        raise ValueError("Column 'sender' not found in DataFrame")

    df_copy = df.copy()

    users = sorted(df_copy["sender"].dropna().unique())
    existing_mapping = load_user_mapping()

    if existing_mapping is None:
        logger.info("No existing mapping found. Creating new mapping.")
        mapping_df = generate_user_mapping(users)
    else:
        mapping_df = existing_mapping.copy()

        existing_users = set(mapping_df["real_name"])
        new_users = [u for u in users if u not in existing_users]

        if new_users:
            logger.info(f"New users detected: {new_users}")

            new_mapping = generate_user_mapping(new_users)
            mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)

    # Save updated mapping
    save_user_mapping(mapping_df)

    mapping_dict = dict(zip(mapping_df["real_name"], mapping_df["pseudo_name"]))
    df_copy["sender"] = df_copy["sender"].map(mapping_dict)

    if df_copy["sender"].isna().any():
        missing = df_copy[df_copy["sender"].isna()]
        raise ValueError(
            f"Anonymization failed. Missing mappings for: "
            f"{missing['sender'].unique()}"
        )

    return df_copy