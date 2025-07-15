import numpy as np
import pandas as pd


def compute_quadratic_fit_r2(series: pd.Series, seasonal_indexer: pd.Series) -> float:
    """
    Fit a quadratic model to the average close price across seasonal units.

    Parameters:
    ----------
    series : pd.Series
        Time series pipeline, indexed by datetime.
    seasonal_indexer : pd.Series
        Seasonal grouping variable (e.g., month or week), same length as series.

    Returns:
    -------
    float
        R-squared of the quadratic fit on seasonal averages.
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

    grouped = df.groupby("group")["y"].mean().reset_index()
    x = grouped["group"].astype(float).values
    y = grouped["y"].values

    if len(x) < 3:
        raise ValueError("Not enough seasonal units to fit a quadratic model.")

    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)
    r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y.mean()) ** 2)

    return r2
