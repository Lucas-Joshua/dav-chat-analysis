import logging
from src import config
from src.modules import (
    data_loader,
    data_cleaning,
    anonymizer,
    visualization,
    feature_engineering,
    metadata,
)

logger = logging.getLogger(__name__)


def prepare_and_run():
    """
    Main pipeline orchestrator.
    """

    logger.info("Pipeline started")

    try:
        df_raw = data_loader.load_raw_chat(
            path=config.RAW_DATA_FILE,
            encoding=config.ENCODING,
        )

        df_clean = data_cleaning.clean_data(df_raw)

        meta = metadata.generate_metadata(df_clean, config.RAW_DATA_FILE)
        metadata.save_metadata(meta)

        df_anonymized = anonymizer.apply_anonymization(df_clean)

        df_final = feature_engineering.apply_all_features(df_anonymized)

        df_final.to_csv(config.CLEAN_CHAT_CSV, index=False)
        df_final.to_parquet(config.CLEAN_CHAT_PARQUET, index=False)

        visualization.create_visuals(df_final)

        logger.info("Pipeline successfully completed")

        return df_final

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise