import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def compute_acf_seasonality(series: pd.Series, lag: int) -> float:
    """
    Compute the ACF value at a given seasonal lag.

    Parameters:
    ----------
    series : pd.Series
        Time series pipeline, indexed by datetime.
    lag : int
        The seasonal lag to evaluate.

    Returns:
    -------
    float
        ACF value at the given lag.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) <= lag:
        raise ValueError(
            f"Series must have more than {lag} "
            f"pipeline points to compute ACF at lag {lag}."
        )

    if np.isclose(np.var(series), 0):
        raise ValueError("Input series must have variance (non-constant values).")

    try:
        acf_vals = acf(series, nlags=lag + 1, fft=True)
        return acf_vals[lag]
    except Exception as e:
        raise RuntimeError(f"ACF computation failed: {e}")
