import logging
import config

import modules.data_loader as data_loader
import modules.data_cleaning as data_cleaning
import modules.visualization as visualization

logger = logging.getLogger(__name__)

def prepare_and_run():
    logger.info("Pipeline gestart")

    # 1. Load raw data
    df_raw = data_loader.load_raw_chat(
        path=config.RAW_DATA_FILE,
        encoding=config.ENCODING,
    )
    logger.info("Raw data geladen")

    # 2. Clean data
    df_clean = data_cleaning.clean_data(df_raw)
    logger.info("Data opgeschoond")

    # 3. Save cleaned data
    csv_path = config.OUTPUT_DIR / "clean_chat.csv"
    parquet_path = config.OUTPUT_DIR / "clean_chat.parquet"

    df_clean.to_csv(csv_path, index=False)
    logger.info(f"CSV opgeslagen: {csv_path}")

    df_clean.to_parquet(parquet_path, index=False)
    logger.info(f"Parquet opgeslagen: {parquet_path}")

    # 4. Visualizations
    visualization.create_visuals(df_clean)
    logger.info("Visualisaties aangemaakt")

    logger.info("Pipeline afgerond")