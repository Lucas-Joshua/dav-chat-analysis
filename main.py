from logging_config import setup_logging
from pipeline import prepare_and_run

def main():
    setup_logging()
    prepare_and_run()

if __name__ == "__main__":
    main()