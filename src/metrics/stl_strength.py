import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def compute_stl_strength(series: pd.Series, period: int, robust: bool = True) -> float:
    """
    Compute STL seasonality strength metric.

    Parameters:
    ----------
    series : pd.Series
        Time series pipeline, indexed by datetime.
    period : int
        The seasonal period (e.g., 12 for monthly pipeline with yearly seasonality).
    robust : bool
        Whether to use robust fitting in STL.

    Returns:
    -------
    strength : float
        Value between 0 and 1 indicating strength of seasonality.
        Higher means more seasonality. Returns np.nan if computation fails.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) < 2 * period:
        raise ValueError(
            f"Series must have at least {2 * period} "
            f"pipeline points for period={period}."
        )

    if np.var(series) == 0:
        raise ValueError("Input series must have variance (non-constant values).")

    try:
        stl = STL(series, period=period, robust=robust)
        result = stl.fit()
        remainder = result.resid
        seasonal = result.seasonal
        strength = max(0, 1 - (np.var(remainder) / np.var(remainder + seasonal)))
        return strength
    except Exception as e:
        raise RuntimeError(f"STL computation failed: {e}")
