# src/pipeline/data_loader.py

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    """
    Class to handle downloading and caching historical financial data from yfinance.
    Loads ticker configuration from a YAML file and manages local caching of data.
    """

    def __init__(
        self,
        config_path: str = "config/tickers_list.yaml",
        use_cache: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the DataLoader.

        Parameters:
        -----------
        config_path : str
            Path to the YAML file containing the list of tickers.
        use_cache : bool
            Whether to use cached files if available.
        verbose : bool
            Whether to print additional logging messages.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = self.project_root / config_path
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = self.project_root / "data" / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> List[str]:
        """
        Loads tickers from the YAML config file
        Returns
        -------
        List: str
            List of ticker symbols
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("tickers", [])

    def _get_cache_filename(
        self, ticker: str, start: str, end: str, interval: str
    ) -> Path:
        """
        Generates a unique cache filename for a ticker based on input parameters
        Parameters
        ----------
        ticker: str
        start: str
        end: str
        interval: str

        Returns
        -------
        Path to cached file
        """
        id_str = f"{ticker}_{start}_{end}_{interval}"
        hash_suffix = hashlib.md5(id_str.encode()).hexdigest()[:8]
        filename = f"{ticker}_{hash_suffix}.parquet"
        return self.cache_dir / filename

    def _download_single_ticker(
        self, ticker: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Downloads data for a single ticker from yfinance with optional caching
        Parameters
        ----------
        ticker: str
        start: str
        end: str
        interval: str

        Returns
        -------
        pd.DataFrame or None
            Downloaded data or None if failed or empty
        """
        cache_path = self._get_cache_filename(ticker, start, end, interval)

        if self.use_cache and cache_path.exists():
            if self.verbose:
                print(f"Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.warning(f"Download failed for {ticker}: {e}")
            return None

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df = (
                df.stack(level=0, future_stack=True)
                .rename_axis(["date", "ticker"])
                .reset_index()
            )
        else:
            df["ticker"] = ticker
            df = df.reset_index()

        df.columns = df.columns.str.lower()
        df["interval"] = interval

        try:
            df.to_parquet(cache_path, index=False)
            if self.verbose:
                print(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {e}")

        return df

    def load(
        self,
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None,
        intervals: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Loads data for all tickers across specified intervals
        Parameters
        ----------
        start_date: str
            Start date for data download
        end_date: str, optional
            End data for data download
        intervals: List[str], optional
            List of time intervals (as in pandas' resample or group by method) to fetch

        Returns
        -------
        pd.DataFrame
            Concatenated data for all tickers and intervals
        """
        if intervals is None:
            intervals = ["1d", "1wk", "1mo"]
        if end_date is None:
            end_date = dt.datetime.today().strftime("%Y-%m-%d")

        tickers = self.load_config()
        all_dfs = []

        for ticker in tickers:
            for interval in intervals:
                if self.verbose:
                    print(f"Loading {ticker} [{interval}]...")
                df = self._download_single_ticker(
                    ticker, start_date, end_date, interval
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)

        if not all_dfs:
            raise ValueError(
                "No data could be loaded for the given tickers and intervals."
            )

        df_all = pd.concat(all_dfs).reset_index(drop=True)
        df_all["date"] = pd.to_datetime(df_all["date"])
        return df_all
