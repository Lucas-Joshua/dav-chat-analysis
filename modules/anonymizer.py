import pandas as pd
import random
import logging
import config

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# =========================================================
# Name list loading
# =========================================================

def load_name_lists():
    first_names = pd.read_csv(config.FIRST_NAMES_FILE)
    last_names = pd.read_csv(config.LAST_NAMES_FILE)

    return (
        first_names.iloc[:, 0].dropna().tolist(),
        last_names.iloc[:, 0].dropna().tolist(),
    )


# =========================================================
# Fake name generation
# =========================================================

def generate_fake_names(n: int):
    random.seed(RANDOM_SEED)

    first_names, last_names = load_name_lists()

    if n > len(first_names) or n > len(last_names):
        raise ValueError("Niet genoeg namen beschikbaar voor unieke combinaties")

    random.shuffle(first_names)
    random.shuffle(last_names)

    return [
        f"{first_names[i]} {last_names[i]}"
        for i in range(n)
    ]


# =========================================================
# Mapping creation / persistence
# =========================================================

def generate_user_mapping(df: pd.DataFrame) -> pd.DataFrame:
    users = sorted(df["sender"].unique())
    fake_names = generate_fake_names(len(users))

    return pd.DataFrame({
        "real_name": users,
        "pseudo_name": fake_names
    })


def save_user_mapping(mapping_df: pd.DataFrame):
    path = config.OUTPUT_DIR / "user_mapping.csv"
    mapping_df.to_csv(path, index=False)
    logger.info(f"User mapping opgeslagen: {path}")


def load_user_mapping():
    path = config.OUTPUT_DIR / "user_mapping.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


# =========================================================
# Public API
# =========================================================

def apply_anonymization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Past anonimisering toe op de sender kolom.
    Mapping wordt persistent opgeslagen.
    """

    mapping_df = load_user_mapping()

    if mapping_df is None:
        logger.info("Nieuwe naam-mapping gegenereerd")
        mapping_df = generate_user_mapping(df)
        save_user_mapping(mapping_df)

    mapping_dict = dict(zip(mapping_df["real_name"], mapping_df["pseudo_name"]))

    df_copy = df.copy()
    df_copy["sender"] = df_copy["sender"].map(mapping_dict)

    return df_copy