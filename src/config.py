"""Project-wide constants and filesystem paths for the data pipeline."""

from pathlib import Path


PROJECT_NAME = "DAV Chat Analysis"
PROJECT_VERSION = "0.1.0"


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


RAW_DATA_FILE = RAW_DIR / "raw_data.txt"

NAMES_DIR = RAW_DIR / "names"
FIRST_NAMES_FILE = NAMES_DIR / "first_names.csv"
LAST_NAMES_FILE = NAMES_DIR / "last_names.csv"


IMG_DIR = BASE_DIR / "img"
LOGS_DIR = BASE_DIR / "logs"

IMG_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


CLEAN_CHAT_CSV = PROCESSED_DIR / "clean_chat_processed.csv"
CLEAN_CHAT_PARQUET = PROCESSED_DIR / "clean_chat_processed.parquet"
USER_MAPPING_FILE = PROCESSED_DIR / "user_mapping.csv"


CSV_SEPARATOR = ";"
ENCODING = "utf-8"

FIGURE_SIZE = (10, 6)
DPI = 300
