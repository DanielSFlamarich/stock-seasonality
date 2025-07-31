# src/pipeline/run_pipeline.py

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# set up basic logging to console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# repo root to sys.path for imports (E402 is disabled because this
# # must run before imports)
# flake8: noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.data_loader import DataLoader
from src.pipeline.seasonality_etl import SeasonalityETL


def run_pipeline(
    config_path: str = "config/tickers_list.yaml",
    start_date: str = "2022-01-01",
    intervals: List[str] = None,
    output_dir: str = "data/.cache",
    frequencies: List[str] = None,
):
    if intervals is None:
        intervals = ["1d"]
    if frequencies is None:
        frequencies = ["W", "ME", "QE", "YE"]

    """
    Runs the full seasonality analysis pipeline, with DataLoader
    to fetch data and SeasonalityETL to compute rolling metrics.

    Parameters:
    ----------
    config_path : str
        Path to YAML file with tickers and metadata.
    start_date : str
        ISO date string for the beginning of data download.
    intervals : List[str]
        Granularity at which to fetch historical price data.
    output_dir : str
        Directory where output CSV with seasonality scores is saved.
    frequencies : List[str]
        Seasonality window frequencies to evaluate. Default: W, ME, QE, YE.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing computed seasonality metrics.
    """
    try:
        logging.info("Loading data...")
        loader = DataLoader(config_path=config_path, use_cache=False, verbose=True)
        df = loader.load(start_date=start_date, intervals=intervals)
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return None

    try:
        logging.info("Running seasonality ETL...")
        etl = SeasonalityETL()
        df_scores = etl.fit_rolling(df, frequencies=frequencies)
        logging.info(f"Metrics computed for {df_scores.shape[0]} rows")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.today().strftime("%Y-%m-%d")
        filename = output_path / f"seasonality_scores_{date_str}.csv"
        df_scores.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")

        return df_scores
    except Exception as e:
        logging.error(f"ETL processing failed: {e}")
        return None


if __name__ == "__main__":
    run_pipeline()
