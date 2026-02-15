import logging
import config

import modules.data_loader as data_loader
import modules.data_cleaning as data_cleaning
import modules.anonymizer as anonymizer
import modules.feature_engineering as feature_engineering
import modules.metadata as metadata
import modules.visualization as visualization

logger = logging.getLogger(__name__)


def prepare_and_run():
    logger.info("Pipeline gestart")

    # =====================================================
    # 1. Load raw data
    # =====================================================
    df_raw = data_loader.load_raw_chat(
        path=config.RAW_DATA_FILE,
        encoding=config.ENCODING,
    )
    logger.info("Raw data geladen")

    # =====================================================
    # 2. Clean data
    # =====================================================
    df_clean = data_cleaning.clean_data(df_raw)
    logger.info("Data opgeschoond")

    # =====================================================
    # 3. Metadata genereren
    # =====================================================
    meta = metadata.generate_metadata(df_clean, config.RAW_DATA_FILE)
    metadata.save_metadata(meta)
    logger.info("Metadata opgeslagen")

    # =====================================================
    # 4. Anonimiseren
    # =====================================================
    df_anonymized = anonymizer.apply_anonymization(df_clean)
    logger.info("Anonimisering toegepast")

    # =====================================================
    # 5. Feature engineering
    # =====================================================
    df_final = feature_engineering.apply_all_features(df_anonymized)
    logger.info("Features toegevoegd")

    # =====================================================
    # 6. Save outputs
    # =====================================================
    csv_path = config.OUTPUT_DIR / "clean_chat.csv"
    parquet_path = config.OUTPUT_DIR / "clean_chat.parquet"

    df_final.to_csv(csv_path, index=False)
    logger.info(f"CSV opgeslagen: {csv_path}")

    df_final.to_parquet(parquet_path, index=False)
    logger.info(f"Parquet opgeslagen: {parquet_path}")

    # =====================================================
    # 7. Visualisaties
    # =====================================================
    visualization.create_visuals(df_final)
    logger.info("Visualisaties aangemaakt")

    logger.info("Pipeline afgerond")