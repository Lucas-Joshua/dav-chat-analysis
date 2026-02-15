from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
NAMES_DIR = DATA_DIR / "names"

FIRST_NAMES_FILE = NAMES_DIR / "first_names.csv"
LAST_NAMES_FILE = NAMES_DIR / "last_names.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_DATA_FILE = DATA_DIR / "raw_data.txt"

# Read settings
CSV_SEPARATOR = ";"
ENCODING = "utf-8"

# Visual settings
FIGURE_SIZE = (10, 6)