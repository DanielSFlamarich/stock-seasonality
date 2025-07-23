import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SeasonalityETL:
    """
    ETL class to compute seasonality metrics from time series financial data.
    Supports both full-series analysis (fit) and rolling-window time evolution
    (fit_rolling).
    TODO: once ready, this should store cleaned up data into db as parquet
    """

    def __init__(
        self,
        seasonal_lags: Optional[Dict[str, int]] = None,
        score_weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
    ):
        """
        Parameters:
        -----------
        seasonal_lags : dict
            Expected seasonality period per interval (e.g. {"1d": 252, "1wk": 52})
        score_weights : dict
            Weights for composite score. Keys: "acf", "p2m", "stl"
        normalize : bool
            Whether to apply min-max normalization before scoring
        """
        self.seasonal_lags = seasonal_lags or {"1d": 252, "1wk": 52, "1mo": 12}
        self.score_weights = score_weights or {"acf": 1 / 3, "p2m": 1 / 3, "stl": 1 / 3}
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

            try:
                acf_vals = acf(series, nlags=lag + 10, fft=True)
                acf_lag_val = acf_vals[lag]
            except ValueError as e:
                logger.warning(f"ACF failed for {ticker}-{interval}: {e}")
                acf_lag_val = np.nan

            try:
                freqs, power = periodogram(series)
                p2m_val = np.max(power) / np.mean(power) if np.mean(power) > 0 else 0
            except ValueError as e:
                logger.warning(f"Periodogram failed for {ticker}-{interval}: {e}")
                p2m_val = np.nan

            try:
                stl = STL(series, period=lag, robust=True)
                result = stl.fit()
                resid_var = np.var(result.resid)
                combined_var = np.var(result.seasonal + result.resid)
                stl_strength = 1 - resid_var / combined_var if combined_var > 0 else 0
            except ValueError as e:
                logger.warning(f"STL failed for {ticker}-{interval}: {e}")
                stl_strength = np.nan

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
        self._compute_scores()

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
        frequencies: List[str] = ["W", "M", "Q", "A"],
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
            Window frequencies (e.g. 'W', 'M', 'Q', 'A')
        min_obs_dict : dict
            Minimum number of observations per window per frequency
        normalize : bool
            Whether to normalize metrics per (interval, freq) before scoring

        Notes:
        ------
        - '1d' is used solely as the base time resolution; we do NOT compute
          seasonality *at* the '1d' level.
        - The metrics (ACF, periodogram, STL) are calculated across time within each
          resampled window (e.g., one score per week or month).

        Returns:
        --------
        DataFrame
            Long-form metrics over time: includes ticker, interval, freq,
            window_start, and scores.
        """
        min_obs_dict = min_obs_dict or {"W": 5, "M": 15, "Q": 40, "A": 200}
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

                    try:
                        acf_vals = acf(series, nlags=min(40, len(series) - 1), fft=True)
                        acf_lag_val = acf_vals[1] if len(acf_vals) > 1 else np.nan
                    except Exception as e:
                        logger.warning(
                            f"ACF failed for {ticker}-{interval} {freq} "
                            f"{window_start}: {e}"
                        )
                        acf_lag_val = np.nan

                    try:
                        freqs_psd, power = periodogram(series)
                        p2m_val = (
                            np.max(power) / np.mean(power) if np.mean(power) > 0 else 0
                        )
                    except Exception as e:
                        logger.warning(
                            f"Periodogram failed for {ticker}-{interval} "
                            f"{freq} {window_start}: {e}"
                        )
                        p2m_val = np.nan

                    try:
                        period = self.seasonal_lags.get(interval, 12)
                        stl = STL(series, period=period, robust=True)
                        result = stl.fit()
                        resid_var = np.var(result.resid)
                        combined_var = np.var(result.seasonal + result.resid)
                        stl_strength = (
                            1 - resid_var / combined_var if combined_var > 0 else 0
                        )
                    except Exception as e:
                        logger.warning(
                            f"STL failed for {ticker}-{interval} {freq} "
                            f"{window_start}: {e}"
                        )
                        stl_strength = np.nan

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

        if normalize and not df_all.empty:
            normed = []
            for (interval, freq), group in df_all.groupby(["interval", "freq"]):
                scaler = MinMaxScaler()
                temp = group.copy()
                try:
                    temp[["acf_lag_val", "p2m_val", "stl_strength"]] = (
                        scaler.fit_transform(
                            temp[["acf_lag_val", "p2m_val", "stl_strength"]]
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"MinMaxScaler failed for interval={interval}, freq={freq}: {e}"
                    )
                    continue
                normed.append(temp)
            df_all = pd.concat(normed, ignore_index=True)

        df_all["seasonality_score_linear"] = (
            self.score_weights["acf"] * df_all["acf_lag_val"]
            + self.score_weights["p2m"] * df_all["p2m_val"]
            + self.score_weights["stl"] * df_all["stl_strength"]
        )

        df_all["seasonality_score_geom"] = (
            df_all["acf_lag_val"] * df_all["p2m_val"] * df_all["stl_strength"]
        ) ** (1 / 3)

        eps = 1e-6
        vals = df_all[["acf_lag_val", "p2m_val", "stl_strength"]].clip(lower=eps)
        df_all["seasonality_score_harmonic"] = 3 / (
            1 / vals["acf_lag_val"] + 1 / vals["p2m_val"] + 1 / vals["stl_strength"]
        )

        self.df_rolling = df_all
        return df_all

    def get_metrics(self) -> pd.DataFrame:
        return self.df_metrics.copy()

    def get_normalized_metrics(self) -> pd.DataFrame:
        return self.df_normalized.copy()

    def get_scores(self) -> pd.DataFrame:
        return self.df_scores.copy()

    def get_rolling_scores(self) -> pd.DataFrame:
        return self.df_rolling.copy() if self.df_rolling is not None else None

    def _compute_scores(self):
        df = self.df_metrics.dropna().copy()

        if self.normalize:
            normed = []
            for interval, group in df.groupby("interval"):
                scaler = MinMaxScaler()
                temp = group.copy()
                temp[["acf_lag_val", "p2m_val", "stl_strength"]] = scaler.fit_transform(
                    temp[["acf_lag_val", "p2m_val", "stl_strength"]]
                )
                normed.append(temp)
            df = pd.concat(normed)
        self.df_normalized = df.copy()

        df["seasonality_score_linear"] = (
            self.score_weights["acf"] * df["acf_lag_val"]
            + self.score_weights["p2m"] * df["p2m_val"]
            + self.score_weights["stl"] * df["stl_strength"]
        )

        df["seasonality_score_geom"] = (
            df["acf_lag_val"] * df["p2m_val"] * df["stl_strength"]
        ) ** (1 / 3)

        eps = 1e-6
        vals = df[["acf_lag_val", "p2m_val", "stl_strength"]].clip(lower=eps)
        df["seasonality_score_harmonic"] = 3 / (
            1 / vals["acf_lag_val"] + 1 / vals["p2m_val"] + 1 / vals["stl_strength"]
        )

        self.df_scores = df
