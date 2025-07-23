import numpy as np
import pandas as pd


def compute_anova_seasonality(series: pd.Series, seasonal_indexer: pd.Series) -> float:
    """
    Compute an ANOVA-like F-statistic for seasonality strength.

    Parameters:
    ----------
    series : pd.Series
        Time series pipeline, indexed by datetime.
    seasonal_indexer : pd.Series
        Categorical grouping (e.g., month, week) same length as series.

    Returns:
    -------
    float
        F-ratio = between-group variance / within-group variance
    """
    if not isinstance(series, pd.Series) or not isinstance(seasonal_indexer, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    if len(series) != len(seasonal_indexer):
        raise ValueError("Series and seasonal_indexer must have the same length.")

    df = pd.DataFrame({"y": series, "group": seasonal_indexer}).dropna()

    if df.empty:
        raise ValueError("Data is empty after removing NA values.")

    group_means = df.groupby("group")["y"].mean()
    overall_mean = df["y"].mean()

    ss_between = sum(df.groupby("group").size() * (group_means - overall_mean) ** 2)
    ss_within = sum((df["y"] - df.groupby("group")["y"].transform("mean")) ** 2)

    if np.isclose(ss_within, 0):
        raise ValueError("Within-group variance is zero, cannot compute F-statistic.")

    return ss_between / ss_within
