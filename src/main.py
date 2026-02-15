from src.logging_config import setup_logging
from src.pipeline import prepare_and_run

def main():
    setup_logging(log_to_file=True)
    prepare_and_run()

if __name__ == "__main__":
    main()