# src/pipeline/data_loader.py

import datetime as dt
import hashlib
import logging
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# constants
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
CACHE_MAX_AGE_DAYS = 7
VALID_INTERVALS = {"1d", "1wk", "1mo", "5d", "1h", "5m", "15m", "30m", "60m", "90m"}


class DataLoader:
    """
    Handles downloading and caching historical financial data from yfinance.

    Features:
    - Exponential backoff retry on transient API failures
    - YAML config validation
    - Date and interval input validation
    - Download statistics tracking
    - Parameter-aware cache (different queries get different cache files)
    - Cache with configurable max age

    Parameters
    ----------
    config_path : str
        Path to YAML config with 'tickers' key
    use_cache : bool
        Whether to use cached parquet data
    verbose : bool
        Enable progress bar and info logging
    save_combined : bool
        Whether to persist combined data to parquet
    combined_cache_path : str or Path, optional
        Override automatic cache path (bypasses parameter-aware naming)
    """

    def __init__(
        self,
        config_path: str = "config/tickers_list.yaml",
        use_cache: bool = True,
        verbose: bool = False,
        save_combined: bool = True,
        combined_cache_path: Optional[Union[str, Path]] = None,
    ):
        # validate config path exists
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not config_file.is_file():
            raise ValueError(f"Config path is not a file: {config_path}")

        self.config_path = config_path
        self.use_cache = use_cache
        self.verbose = verbose
        self.save_combined = save_combined
        self.stats: Dict[str, int] = {"success": 0, "failed": 0, "total": 0}

        self.cache_dir = Path("data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # if explicitly provided, lock cache path; otherwise compute per-query in load()
        self._cache_path_override = (
            Path(combined_cache_path) if combined_cache_path else None
        )
        self.combined_cache_path: Optional[Path] = self._cache_path_override

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def _read_tickers(self) -> List[str]:
        """
        Read and validate tickers from YAML config.

        Returns
        -------
        List[str]
            Validated list of ticker symbols

        Raises
        ------
        ValueError
            If YAML schema is invalid or tickers list is empty
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dict, got {type(config)}")
        if "tickers" not in config:
            raise ValueError("Config must contain 'tickers' key")
        if not isinstance(config["tickers"], list):
            raise ValueError(f"'tickers' must be a list, got {type(config['tickers'])}")
        if not all(isinstance(t, str) for t in config["tickers"]):
            raise ValueError("All tickers must be strings")
        if len(config["tickers"]) == 0:
            raise ValueError("Tickers list cannot be empty")

        return config["tickers"]

    @staticmethod
    def _validate_date(date_str: str) -> None:
        """
        Validate ISO date format (YYYY-MM-DD).
        """
        try:
            dt.datetime.fromisoformat(date_str)
        except ValueError:
            raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")

    @staticmethod
    def _validate_intervals(intervals: List[str]) -> None:
        """
        Validate that all intervals are supported by yfinance.
        """
        invalid = set(intervals) - VALID_INTERVALS
        if invalid:
            raise ValueError(
                f"Invalid intervals: {invalid}. Valid: {sorted(VALID_INTERVALS)}"
            )

    @staticmethod
    def _compute_cache_hash(
        tickers: List[str],
        intervals: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        """
        Compute a short hash for cache key based on query parameters.

        Includes tickers, intervals, and date range so different queries
        never collide. Order-independent for tickers and intervals.
        """
        key = (
            "+".join(sorted(tickers))
            + "|"
            + "+".join(sorted(intervals))
            + "|"
            + start_date
            + "|"
            + end_date
        )
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _build_cache_path(
        self,
        tickers: List[str],
        intervals: List[str],
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Build a cache filename that encodes the query parameters.

        Format: all_data_{start}_{end}_{hash}.parquet
        Example: all_data_2012-01-01_2023-01-01_a1b2c3d4.parquet
        """
        h = self._compute_cache_hash(tickers, intervals, start_date, end_date)
        return self.cache_dir / f"all_data_{start_date}_{end_date}_{h}.parquet"

    def _is_cache_fresh(self) -> bool:
        """Check if cache file exists and is younger than CACHE_MAX_AGE_DAYS."""
        if self.combined_cache_path is None or not self.combined_cache_path.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(
            self.combined_cache_path.stat().st_mtime
        )
        return age.days < CACHE_MAX_AGE_DAYS

    def _download_with_retry(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """
        Download ticker data with exponential backoff retry.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., 'AAPL', 'SAP.DE')
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval (e.g., '1d', '1wk')

        Returns
        -------
        pd.DataFrame or None
            Raw OHLCV data, or None if all retries exhausted
        """
        for attempt in range(MAX_RETRIES):
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
                    return df

                # empty result --> retry (ticker may be temporarily unavailable)
                logger.debug(
                    f"Empty result for {ticker} [{interval}], "
                    f"attempt {attempt + 1}/{MAX_RETRIES}"
                )

            except Exception as e:
                logger.warning(
                    f"Download failed for {ticker} [{interval}], "
                    f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                )

            # exponential backoff (skip sleep on last attempt)
            if attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.debug(f"Retrying {ticker} in {wait:.1f}s...")
                time.sleep(wait)

        logger.warning(f"All {MAX_RETRIES} attempts failed for {ticker} [{interval}]")
        return None

    def load(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        intervals: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load historical data for all configured tickers.

        Downloads from yfinance (with retry) or returns cached data.
        Cache files are keyed by (tickers, intervals, start_date, end_date)
        so different queries never collide.

        After completion, download statistics are available via self.stats.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date (defaults to today)
        intervals : List[str], optional
            Data granularity (defaults to ['1d'])

        Returns
        -------
        pd.DataFrame
            Combined OHLCV data with 'ticker' and 'interval' columns

        Raises
        ------
        ValueError
            If no data could be loaded for any ticker
        """
        if intervals is None:
            intervals = ["1d"]

        # Validate inputs
        self._validate_date(start_date)
        if end_date:
            self._validate_date(end_date)
        self._validate_intervals(intervals)

        # resolve end_date early so cache key is deterministic
        end_date = end_date or dt.datetime.today().strftime("%Y-%m-%d")

        # read tickers for cache key computation
        tickers = self._read_tickers()

        # build parameter-aware cache path (unless overridden in __init__)
        if self._cache_path_override is None:
            self.combined_cache_path = self._build_cache_path(
                tickers, intervals, start_date, end_date
            )

        # check cache
        if self.use_cache and self._is_cache_fresh():
            self._log(f"Loading cached data from {self.combined_cache_path}")
            return pd.read_parquet(self.combined_cache_path)

        all_dfs: List[pd.DataFrame] = []

        # reset stats
        pairs = list(product(tickers, intervals))
        self.stats = {"success": 0, "failed": 0, "total": len(pairs)}

        for ticker, interval in tqdm(
            pairs, desc="Downloading data", disable=not self.verbose
        ):
            df = self._download_with_retry(ticker, start_date, end_date, interval)

            if df is not None and not df.empty:
                # flatten any nested tuples or multi-indexes defensively
                df.columns = [
                    col[0] if isinstance(col, (tuple, list)) else col
                    for col in df.columns
                ]
                df = df.reset_index()
                df["ticker"] = ticker
                df["interval"] = interval
                all_dfs.append(df)
                self.stats["success"] += 1
            else:
                self.stats["failed"] += 1

        # report statistics
        success_rate = (
            100 * self.stats["success"] / self.stats["total"]
            if self.stats["total"] > 0
            else 0
        )
        self._log(
            f"Download complete: {self.stats['success']}/{self.stats['total']} "
            f"({success_rate:.0f}% success)"
        )
        if success_rate < 50:
            logger.warning(
                f"Low success rate: {success_rate:.0f}%. "
                f"Check network or ticker validity."
            )

        if not all_dfs:
            raise ValueError("No data could be loaded for any ticker or interval.")

        df_all = pd.concat(all_dfs, ignore_index=True)

        # ensure all column names are strings before lower casing
        df_all.columns = [str(col).lower() for col in df_all.columns]

        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"])
        else:
            raise ValueError("Expected column 'date' not found after standardization.")

        if self.save_combined:
            df_all.to_parquet(self.combined_cache_path)
            self._log(f"Saved combined data to {self.combined_cache_path}")

        return df_all
