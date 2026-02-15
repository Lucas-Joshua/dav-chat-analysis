import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)$"
)

# Alles wat WhatsApp vaak injecteert aan "rare" whitespace / direction marks
_INVISIBLE = [
    "\u202f",  # narrow no-break space
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\ufeff",  # BOM
]

def normalize_sender(sender: str) -> str:
    sender = "" if sender is None else str(sender)

    # Verwijder leading "~" (WhatsApp systeem/rol prefix) en normaliseer spaties
    sender = sender.replace("~", "")

    # Verwijder onzichtbare unicode tekens
    for ch in _INVISIBLE:
        sender = sender.replace(ch, "")

    # NBSP -> normale spatie, en whitespace normaliseren
    sender = sender.replace("\xa0", " ")
    sender = " ".join(sender.split())

    return sender.strip()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Start opschonen van data")

    messages = []
    current = None

    for line in df.iloc[:, 0]:
        line = "" if line is None else str(line).rstrip("\n")

        m = MESSAGE_PATTERN.match(line.strip())
        if m:
            if current:
                messages.append(current)

            date, time, sender, msg = m.groups()
            current = {
                "datetime": f"{date} {time}",
                "sender": normalize_sender(sender),
                "message": msg.strip(),
            }
        else:
            if current and line.strip():
                current["message"] += "\n" + line.strip()

    if current:
        messages.append(current)

    df_clean = pd.DataFrame(messages)
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"], format="%d-%m-%Y %H:%M:%S")

    # Extra safety: normaliseer sender nogmaals (voor het geval)
    df_clean["sender"] = df_clean["sender"].astype(str).map(normalize_sender)

    logger.info(f"Opschonen afgerond ({len(df_clean)} berichten)")
    return df_clean