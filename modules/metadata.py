import logging
import re
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# Unicode control characters die WhatsApp gebruikt
WHATSAPP_CONTROL_CHARS = [
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u202a",
    "\u202c",
]


def contains_control_char(text: str) -> bool:
    """
    Detecteert of tekst WhatsApp control characters bevat.
    """
    return any(char in text for char in WHATSAPP_CONTROL_CHARS)


def has_real_text(text: str) -> bool:
    """
    Controleert of bericht echte inhoud bevat (letters of cijfers).
    """
    return bool(re.search(r"[A-Za-z0-9]", text))


def detect_group_name(df: pd.DataFrame) -> str | None:
    """
    Detecteert groepsnaam op basis van eerste regel in export.
    """
    if df.empty:
        return None
    return df.iloc[0]["sender"]


def is_system_message(row, group_name: str) -> bool:
    """
    Robuuste detectie van WhatsApp systeembericht.
    Volledig losgekoppeld van taal.
    """

    sender = str(row["sender"])
    message = str(row["message"])

    # 1. WhatsApp control characters aanwezig
    if contains_control_char(message):
        return True

    # 2. Geen echte tekst
    if not has_real_text(message):
        return True

    # 3. Sender is groepsnaam EN bericht bevat geen normale conversatie
    if sender == group_name and len(message.split()) < 5:
        return True

    return False


def detect_chat_start(df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Detecteert eerste echte gebruikersactiviteit.
    """

    if df.empty:
        return None

    group_name = detect_group_name(df)

    df_filtered = df.copy()

    df_filtered["is_system"] = df_filtered.apply(
        lambda row: is_system_message(row, group_name),
        axis=1,
    )

    df_filtered = df_filtered[~df_filtered["is_system"]]

    if df_filtered.empty:
        return None

    return df_filtered["datetime"].min()


def generate_metadata(df: pd.DataFrame, raw_file_path: str) -> dict:
    """
    Genereert metadata over de chat.
    """

    raw_start = df["datetime"].min() if not df.empty else None
    real_start = detect_chat_start(df)

    metadata = {
        "file_name": Path(raw_file_path).name,
        "total_messages": len(df),
        "raw_start_date": str(raw_start) if raw_start else None,
        "real_start_date": str(real_start) if real_start else None,
    }

    logger.info("Metadata gegenereerd")
    return metadata


def save_metadata(metadata: dict):
    """
    Slaat metadata op als JSON in output directory.
    """

    import json
    import config

    path = config.OUTPUT_DIR / "metadata.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"Metadata opgeslagen: {path}")