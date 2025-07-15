import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SeasonalityETL:
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
            Mapping of interval to seasonal lag (e.g., {"1d": 252})
        score_weights : dict
            Weights for metrics in linear combination. Keys: "acf", "p2m", "stl"
        normalize : bool
            Whether to min-max normalize the raw metric scores before scoring
        """
        self.seasonal_lags = seasonal_lags or {"1d": 252, "1wk": 52, "1mo": 12}
        self.score_weights = score_weights or {"acf": 1 / 3, "p2m": 1 / 3, "stl": 1 / 3}
        self.normalize = normalize
        self.df_metrics = None
        self.df_normalized = None
        self.df_scores = None

    def fit(
        self, df: pd.DataFrame, return_stage: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Run the ETL: compute metrics and scores for grouped time series.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: ["date", "close", "ticker", "interval"]
        return_stage : str or None
            One of: 'metrics', 'normalized', 'scores'. If set, returns
            the DataFrame at that stage.
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

    def get_metrics(self) -> pd.DataFrame:
        return self.df_metrics.copy()

    def get_normalized_metrics(self) -> pd.DataFrame:
        return self.df_normalized.copy()

    def get_scores(self) -> pd.DataFrame:
        return self.df_scores.copy()
