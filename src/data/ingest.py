import pandas as pd
from pathlib import Path

from src.utils.config import load_params
from src.utils.logger import logger


def ingest_data():
    """Read the raw NYC taxi dataset."""

    params = load_params()

    data_path = Path(params["dataset"]["raw_data_path"])

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)

    logger.info("========== DATA INGESTION ==========")
    logger.info(f"Dataset Path : {data_path}")
    logger.info(f"Rows         : {df.shape[0]}")
    logger.info(f"Columns      : {df.shape[1]}")
    logger.info(f"Memory Usage : {round(df.memory_usage(deep=True).sum()/1024/1024,2)} MB")
    logger.info("====================================")

    return df


if __name__ == "__main__":
    dataframe = ingest_data()
    print(dataframe.head())