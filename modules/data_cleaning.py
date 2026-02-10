import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

# Regex voor WhatsApp-berichten
MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)$"
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schoont WhatsApp chat-export data op en structureert deze
    naar datetime, sender en message.
    """
    logger.info("Start opschonen van data")

    messages = []
    current_message = None

    for line in df.iloc[:, 0]:
        line = str(line).strip()

        match = MESSAGE_PATTERN.match(line)
        if match:
            # nieuw bericht
            if current_message:
                messages.append(current_message)

            date, time, sender, message = match.groups()
            current_message = {
                "datetime": f"{date} {time}",
                "sender": sender.replace("~", "").strip(),
                "message": message.strip(),
            }
        else:
            # multiline bericht → append
            if current_message:
                current_message["message"] += " " + line

    # laatste bericht toevoegen
    if current_message:
        messages.append(current_message)

    df_clean = pd.DataFrame(messages)

    # datetime conversie
    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"], format="%d-%m-%Y %H:%M:%S"
    )

    logger.info(f"Opschonen afgerond ({len(df_clean)} berichten)")
    return df_clean