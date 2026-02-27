# src/pipeline/seasonality_etl.py

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from src.scoring.meta_scores import add_meta_scores, normalize_metrics_by_group

logger = logging.getLogger(__name__)

# constants for data validation and computation
MIN_SERIES_BUFFER = 10  # extra observations needed beyond seasonal lag
ACF_EXTRA_LAGS = 10  # extra lags to compute for ACF beyond target lag
DEFAULT_MIN_OBS = {
    "W": 5,  # Weekly: at least 5 days
    "ME": 15,  # Monthly: at least 15 days
    "QE": 40,  # Quarterly: at least 40 days
    "YE": 200,  # Yearly: at least 200 days
}
MIN_STL_OBSERVATIONS = 2  # minimum ratio: len(series) / period >= 2


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
        + High value (close to 1): the patterns repeats reliably from one
          period to the other.
        + Low value (close to 0): little or no repeatability in the gap.
        + Simple and intuitive, can be thrown off by trends or regime changes.
    - P2M (Peak-to-mean ratio from periodogram)
        + Checks frequency and asks; is there a stand out repeating rhythm?
        + High value: (sharp peaks, signal >>> noise)
        + Low value: (no peaks = noisy series)
        + Good for crisp cycles, sensitive to short stories or heavy noise
    - STL (Seasonal strength from STL decomposition)
        + Split into trend + seasonal + remainder (noise) and measure how
          much of the variation
          is explained by the seasonal part
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
    ) -> None:
        """
        Initialize SeasonalityETL with configuration.

        Parameters
        ----------
        seasonal_lags : dict, optional
            Expected seasonality period per interval (e.g. {"1d": 252, "1wk": 52})
            Defaults to {"1d": 252, "1wk": 52, "1mo": 12}
        normalize : bool, default True
            Whether to apply min-max normalization before scoring
        """
        self.seasonal_lags = seasonal_lags or {"1d": 252, "1wk": 52, "1mo": 12}
        self.normalize = normalize

        self.df_metrics: Optional[pd.DataFrame] = None
        self.df_normalized: Optional[pd.DataFrame] = None
        self.df_scores: Optional[pd.DataFrame] = None
        self.df_rolling: Optional[pd.DataFrame] = None

    def fit(
        self, df: pd.DataFrame, return_stage: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Batch computation of seasonality metrics over entire series per ticker/interval.

        Parameters
        ----------
        df : DataFrame
            Must include: ["date", "close", "ticker", "interval"]
        return_stage : str, optional
            If provided, returns one of: 'metrics', 'normalized', 'scores'
            If None, stores results but returns None

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with requested stage, or None

        Raises
        ------
        ValueError
            If required columns are missing from df
        """
        # Validate input DataFrame
        required_cols = ["date", "close", "ticker", "interval"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing_cols}"
            )

        if df.empty:
            logger.warning("Input DataFrame is empty, returning empty results")
            self.df_metrics = pd.DataFrame()
            self.df_normalized = pd.DataFrame()
            self.df_scores = pd.DataFrame()
            return self._get_stage_result(return_stage)

        results = []

        for (ticker, interval), df_group in df.groupby(["ticker", "interval"]):
            series = df_group.sort_values("date")["close"].dropna()
            lag = self.seasonal_lags.get(interval, 12)

            # Use constant instead of magic number
            if len(series) < lag + MIN_SERIES_BUFFER:
                logger.debug(
                    f"Skipping {ticker}[{interval}]: insufficient data "
                    f"({len(series)} < {lag + MIN_SERIES_BUFFER})"
                )
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

        return self._get_stage_result(return_stage)

    def _get_stage_result(self, return_stage: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Helper method to return appropriate stage result.

        Parameters
        ----------
        return_stage : str, optional
            One of: 'metrics', 'normalized', 'scores', or None

        Returns
        -------
        Optional[pd.DataFrame]
            Requested stage DataFrame or None
        """
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

        Parameters
        ----------
        df : DataFrame
            Must include: ["date", "close", "ticker", "interval"]
        frequencies : list of str, default ["W", "ME", "QE", "YE"]
            Window frequencies (e.g. 'W', 'ME', 'QE', 'YE')
        min_obs_dict : dict, optional
            Minimum number of observations per window per frequency
            Defaults to DEFAULT_MIN_OBS constant
        normalize : bool, optional
            Whether to normalize metrics per (interval, freq) before scoring
            If None, uses instance default

        Returns
        -------
        pd.DataFrame
            Long-form metrics over time with meta-scores

        Raises
        ------
        ValueError
            If required columns are missing from df
        """
        # Validate input DataFrame
        required_cols = ["date", "close", "ticker", "interval"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing_cols}"
            )

        if df.empty:
            logger.warning("Input DataFrame is empty, returning empty results")
            return pd.DataFrame()

        # Use constant instead of hardcoded dict
        min_obs_dict = min_obs_dict or DEFAULT_MIN_OBS
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
                        f"Resampling failed for {ticker}[{interval}] freq={freq}: {e}"
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

        metric_cols = ["acf_lag_val", "p2m_val", "stl_strength"]
        df_all = pd.DataFrame(results).dropna(subset=metric_cols, how="all")

        # use shared normalization and scoring
        if normalize and not df_all.empty:
            df_all = normalize_metrics_by_group(
                df_all,
                group_cols=["interval", "freq"],
                metric_cols=["acf_lag_val", "p2m_val", "stl_strength"],
            )

        # use shared meta-scoring (replaces duplicated logic)
        if not df_all.empty:
            df_all = add_meta_scores(df_all)

        self.df_rolling = df_all
        return df_all

    # private methods for metric computation
    def _compute_acf(self, series: pd.Series, lag: int) -> float:
        """
        Compute ACF at given lag with error handling.

        Parameters
        ----------
        series : pd.Series
            Time series data
        lag : int
            Lag at which to compute autocorrelation

        Returns
        -------
        float
            ACF value at specified lag, or np.nan if computation fails
        """
        try:
            # Use constant instead of magic number
            nlags = min(lag + ACF_EXTRA_LAGS, len(series) - 1)
            acf_vals = acf(series, nlags=nlags, fft=True)
            return float(acf_vals[min(lag, len(acf_vals) - 1)])
        except Exception as e:
            logger.warning(
                "ACF computation failed for series length=%s, lag=%s: %s",
                len(series),
                lag,
                e,
            )
            return np.nan

    def _compute_p2m(self, series: pd.Series) -> float:
        """
        Compute peak-to-mean ratio from periodogram.

        Parameters
        ----------
        series : pd.Series
            Time series data

        Returns
        -------
        float
            Ratio of max power to mean power, or np.nan if computation fails
        """
        try:
            _, power = periodogram(series)
            mean_power = np.mean(power)
            result = float(np.max(power) / mean_power if mean_power > 0 else 0)
            return result
        except Exception as e:
            logger.warning(
                f"P2M computation failed for series length={len(series)}: {e}"
            )
            return np.nan

    def _compute_stl_strength(self, series: pd.Series, period: int) -> float:
        """
        Compute STL seasonal strength with period validation.

        STL decomposition requires at least 2 complete periods of data.
        This validates the constraint before attempting decomposition.

        Parameters
        ----------
        series : pd.Series
            Time series data
        period : int
            Seasonal period (e.g., 252 for daily data with yearly seasonality)

        Returns
        -------
        float
            Seasonal strength (0 to 1), or np.nan if computation fails or
            insufficient data for the given period
        """
        try:
            # CRITICAL VALIDATION: STL requires len(series) >= 2 * period
            # This prevents cryptic errors from statsmodels
            if len(series) < MIN_STL_OBSERVATIONS * period:
                logger.debug(
                    "Insufficient data for STL: len=%s < %s (period=%s)",
                    len(series),
                    MIN_STL_OBSERVATIONS * period,
                    period,
                )
                return np.nan

            # Additional safety: period must be odd for STL
            # If even, increment by 1
            stl_period = period if period % 2 == 1 else period + 1

            stl = STL(series, period=stl_period, robust=True)
            result = stl.fit()
            resid_var = np.var(result.resid)
            combined_var = np.var(result.seasonal + result.resid)
            strength = float(1 - resid_var / combined_var if combined_var > 0 else 0)
            return strength
        except Exception as e:
            logger.warning(
                "STL computation failed for series length=%s, period=%s: %s",
                len(series),
                period,
                e,
            )
            return np.nan

    def get_metrics(self) -> Optional[pd.DataFrame]:
        """
        Return raw metrics DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            Raw metrics or None if fit() has not been called
        """
        return self.df_metrics.copy() if self.df_metrics is not None else None

    def get_normalized_metrics(self) -> Optional[pd.DataFrame]:
        """
        Return normalized metrics DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            Normalized metrics or None if not available
        """
        return self.df_normalized.copy() if self.df_normalized is not None else None

    def get_scores(self) -> Optional[pd.DataFrame]:
        """
        Return scored DataFrame with meta-scores.

        Returns
        -------
        Optional[pd.DataFrame]
            Scored DataFrame or None if fit() has not been called
        """
        return self.df_scores.copy() if self.df_scores is not None else None

    def get_rolling_scores(self) -> Optional[pd.DataFrame]:
        """
        Return rolling window scores DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            Rolling scores or None if fit_rolling() has not been called
        """
        return self.df_rolling.copy() if self.df_rolling is not None else None

    def _compute_scores(self) -> None:
        """
        Internal method to compute scores from raw metrics.

        Uses shared meta_scores module for normalization and scoring.
        Updates self.df_normalized and self.df_scores in place.
        """
        if self.df_metrics is None or self.df_metrics.empty:
            self.df_normalized = pd.DataFrame()
            self.df_scores = pd.DataFrame()
            return

        df = self.df_metrics.dropna().copy()

        # Protect against all-NaN metrics after dropna
        if df.empty:
            logger.warning("All metrics are NaN after dropna, returning empty results")
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
