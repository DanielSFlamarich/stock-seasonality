# src/pipeline/seasonality_etl.py (UPDATED)
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from src.scoring.meta_scores import add_meta_scores, normalize_metrics_by_group

logger = logging.getLogger(__name__)


class SeasonalityETL:
    """
    ETL class to compute seasonality metrics from time series financial data.

    Supports:
    - Full-series analysis (fit)
    - Rolling-window time evolution (fit_rolling)

    Metrics:
    --------
    - ACF (Auto-correlation at seasonal lag)
        + "Echo-score" where we compare the series to itself shifted one full season
        + High value (close to 1): the patterns repeats reliably from
          one period to the other.
        + Low value (close to 0): little or no repeatability in the gap.
        + Simple and intuitive, can be thrown off by trends or regime changes.
    - P2M (Peak-to-mean ratio from periodogram)
        + Checks frequency and asks; is there a stand out repeating rhythm?
        + High value: (sharp peaks, signal >>> noise)
        + Low value: (no peaks = noisy series)
        + Good for crisp cycles, sensitive to short stories or heavy noise
    - STL (Seasonal strength from STL decomposition)
        + Split into trend + seasonal + remainder (noise) and measure how
          much of the variation is explained by the seasonal part
        + High value: most var is systematic and seasonal (patter explains data well)
        + Low value: randomness or noise dominate
        + Robust to trend, needs good volume of data to fit properly

    Meta-scores:
    ------------
    Linear, geometric, and harmonic combinations via src.scoring.meta_scores
    """

    def __init__(
        self,
        seasonal_lags: Optional[Dict[str, int]] = None,
        normalize: bool = True,
    ):
        """
        Parameters:
        -----------
        seasonal_lags : dict
            Expected seasonality period per interval (e.g. {"1d": 252, "1wk": 52})
        normalize : bool
            Whether to apply min-max normalization before scoring
        """
        self.seasonal_lags = seasonal_lags or {"1d": 252, "1wk": 52, "1mo": 12}
        self.normalize = normalize

        self.df_metrics = None
        self.df_normalized = None
        self.df_scores = None
        self.df_rolling = None

    def fit(
        self, df: pd.DataFrame, return_stage: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Batch computation of seasonality metrics over entire series per ticker/interval.

        Parameters:
        -----------
        df : DataFrame
            Must include: ["date", "close", "ticker", "interval"]
        return_stage : str or None
            If provided, returns one of: 'metrics', 'normalized', 'scores'
        """
        results = []

        for (ticker, interval), df_group in df.groupby(["ticker", "interval"]):
            series = df_group.sort_values("date")["close"].dropna()
            lag = self.seasonal_lags.get(interval, 12)

            if len(series) < lag + 10:
                continue

            # compute metrics using private methods for clarity
            acf_lag_val = self._compute_acf(series, lag)
            p2m_val = self._compute_p2m(series)
            stl_strength = self._compute_stl_strength(series, lag)

            results.append(
                {
                    "ticker": ticker,
                    "interval": interval,
                    "acf_lag_val": acf_lag_val,
                    "p2m_val": p2m_val,
                    "stl_strength": stl_strength,
                }
            )

        self.df_metrics = pd.DataFrame(results)
        self._compute_scores()  # still uses shared module internally

        if return_stage == "metrics":
            return self.get_metrics()
        elif return_stage == "normalized":
            return self.get_normalized_metrics()
        elif return_stage == "scores":
            return self.get_scores()
        return None

    def fit_rolling(
        self,
        df: pd.DataFrame,
        frequencies: List[str] = ["W", "ME", "QE", "YE"],
        min_obs_dict: Optional[Dict[str, int]] = None,
        normalize: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Compute seasonality metrics over calendar-aligned rolling windows.

        Parameters:
        -----------
        df : DataFrame
            Must include: ["date", "close", "ticker", "interval"]
        frequencies : list of str
            Window frequencies (e.g. 'W', 'ME', 'QE', 'YE')
        min_obs_dict : dict
            Minimum number of observations per window per frequency
        normalize : bool
            Whether to normalize metrics per (interval, freq) before scoring

        Returns:
        --------
        DataFrame
            Long-form metrics over time with meta-scores.
        """
        min_obs_dict = min_obs_dict or {"W": 5, "ME": 15, "QE": 40, "YE": 200}
        normalize = self.normalize if normalize is None else normalize
        results = []

        for (ticker, interval), df_group in df.groupby(["ticker", "interval"]):
            df_group = df_group.dropna(subset=["date", "close"])
            df_group = df_group.sort_values("date").set_index("date")

            for freq in frequencies:
                try:
                    windows = df_group["close"].resample(freq)
                except Exception as e:
                    logger.warning(
                        f"Resampling failed for {ticker}-{interval} freq={freq}: {e}"
                    )
                    continue

                for window_start, series in windows:
                    series = series.dropna()
                    if len(series) < min_obs_dict.get(freq, 10):
                        continue

                    # use private methods for computation
                    acf_lag_val = self._compute_acf(series, lag=1)
                    p2m_val = self._compute_p2m(series)
                    stl_strength = self._compute_stl_strength(
                        series, period=self.seasonal_lags.get(interval, 12)
                    )

                    results.append(
                        {
                            "ticker": ticker,
                            "interval": interval,
                            "freq": freq,
                            "window_start": window_start,
                            "acf_lag_val": acf_lag_val,
                            "p2m_val": p2m_val,
                            "stl_strength": stl_strength,
                        }
                    )

        df_all = pd.DataFrame(results).dropna()

        # Early return if no windows survived filtering
        if df_all.empty:
            self.df_rolling = df_all
            return df_all

        # use shared normalization and scoring
        if normalize:
            df_all = normalize_metrics_by_group(
                df_all,
                group_cols=["interval", "freq"],
                metric_cols=["acf_lag_val", "p2m_val", "stl_strength"],
            )

        # use shared meta-scoring (replaces duplicated logic)
        df_all = add_meta_scores(df_all)

        self.df_rolling = df_all
        return df_all

    # private methods for metric computation
    def _compute_acf(self, series: pd.Series, lag: int) -> float:
        """
        Compute ACF at given lag with error handling.
        """
        try:
            acf_vals = acf(series, nlags=min(lag + 10, len(series) - 1), fft=True)
            return float(acf_vals[min(lag, len(acf_vals) - 1)])
        except Exception as e:
            logger.warning(f"ACF computation failed: {e}")
            return np.nan

    def _compute_p2m(self, series: pd.Series) -> float:
        """
        Compute peak-to-mean ratio from periodogram.
        """
        try:
            _, power = periodogram(series)
            mean_power = np.mean(power)
            return float(np.max(power) / mean_power if mean_power > 0 else 0)
        except Exception as e:
            logger.warning(f"P2M computation failed: {e}")
            return np.nan

    def _compute_stl_strength(self, series: pd.Series, period: int) -> float:
        """
        Compute STL seasonal strength.
        """
        try:
            stl = STL(series, period=period, robust=True)
            result = stl.fit()
            resid_var = np.var(result.resid)
            combined_var = np.var(result.seasonal + result.resid)
            return float(1 - resid_var / combined_var if combined_var > 0 else 0)
        except Exception as e:
            logger.warning(f"STL computation failed: {e}")
            return np.nan

    def get_metrics(self) -> pd.DataFrame:
        """
        Return raw metrics DataFrame.
        """
        return self.df_metrics.copy() if self.df_metrics is not None else None

    def get_normalized_metrics(self) -> pd.DataFrame:
        """
        Return normalized metrics DataFrame.
        """
        return self.df_normalized.copy() if self.df_normalized is not None else None

    def get_scores(self) -> pd.DataFrame:
        """
        Return scored DataFrame.
        """
        return self.df_scores.copy() if self.df_scores is not None else None

    def get_rolling_scores(self) -> pd.DataFrame:
        """
        Return rolling window scores DataFrame.
        """
        return self.df_rolling.copy() if self.df_rolling is not None else None

    def _compute_scores(self):
        """
        Internal method to compute scores from raw metrics.
        Now uses shared meta_scores module to avoid duplication.
        """
        df = self.df_metrics.dropna().copy()

        # Early return if no data (prevents KeyError in normalize_metrics_by_group)
        if df.empty:
            self.df_normalized = pd.DataFrame()
            self.df_scores = pd.DataFrame()
            return

        # Normalize per interval if requested
        if self.normalize:
            df = normalize_metrics_by_group(
                df,
                group_cols=["interval"],
                metric_cols=["acf_lag_val", "p2m_val", "stl_strength"],
            )

        self.df_normalized = df.copy()

        # Use shared scoring module
        df = add_meta_scores(df)

        self.df_scores = df
