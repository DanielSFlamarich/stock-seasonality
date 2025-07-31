# src/pipeline/data_loader.py

import datetime as dt
import logging
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: last day (end_date) should be current date


class DataLoader:
    """
    Handles downloading and caching historical financial data from yfinance.
    Caches the entire dataset to a single parquet file with daystamp.
    """

    def __init__(
        self,
        config_path: str = "config/tickers_list.yaml",
        use_cache: bool = True,
        verbose: bool = False,
        save_combined: bool = True,
        combined_cache_path: Optional[Union[str, Path]] = None,
    ):
        self.config_path = config_path
        self.use_cache = use_cache
        self.verbose = verbose
        self.save_combined = save_combined

        self.cache_dir = Path("data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if combined_cache_path:
            self.combined_cache_path = Path(combined_cache_path)
        else:
            today_str = dt.datetime.today().strftime("%Y%m%d")
            self.combined_cache_path = self.cache_dir / f"all_data_{today_str}.parquet"

    def _log(self, msg):
        if self.verbose:
            logger.info(msg)

    def _read_tickers(self) -> List[str]:
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["tickers"]

    def load(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        intervals: List[str] = ["1d"],
    ) -> pd.DataFrame:
        if self.use_cache and self.combined_cache_path.exists():
            self._log(f"Loading cached data from {self.combined_cache_path}")
            return pd.read_parquet(self.combined_cache_path)

        tickers = self._read_tickers()
        end_date = end_date or dt.datetime.today().strftime("%Y-%m-%d")
        all_dfs = []

        pairs = list(product(tickers, intervals))
        for ticker, interval in tqdm(
            pairs, desc="Downloading data", disable=not self.verbose
        ):
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if df is not None and not df.empty:
                    # Flatten any nested tuples or multi-indexes defensively
                    df.columns = [
                        col[0] if isinstance(col, (tuple, list)) else col
                        for col in df.columns
                    ]

                    df = df.reset_index()
                    df["ticker"] = ticker
                    df["interval"] = interval
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {ticker} [{interval}]: {e}")

        if not all_dfs:
            raise ValueError("No data could be loaded for any ticker or interval.")

        df_all = pd.concat(all_dfs, ignore_index=True)

        # Ensure all column names are strings before lowercasing
        df_all.columns = [str(col).lower() for col in df_all.columns]

        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"])
        else:
            raise ValueError("Expected column 'date' not found after standardization.")

        if self.save_combined:
            df_all.to_parquet(self.combined_cache_path)
            self._log(f"Saved combined data to {self.combined_cache_path}")

        return df_all
