import numpy as np
import pandas as pd
from scipy.signal import periodogram


def compute_periodogram_strength(series: pd.Series) -> float:
    """
    Compute the peak-to-mean ratio from the periodogram of the time series.

    Parameters:
    ----------
    series : pd.Series
        Time series pipeline, indexed by datetime.

    Returns:
    -------
    float
        Peak-to-mean power ratio from the periodogram.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) < 20:
        raise ValueError(
            "Series must have at least 20 pipeline points for periodogram analysis."
        )

    if np.isclose(np.var(series), 0):
        raise ValueError("Input series must have variance (non-constant values).")

    try:
        freqs, power = periodogram(series)
        if np.isclose(power.mean(), 0):
            return 0.0
        return power.max() / power.mean()
    except Exception as e:
        raise RuntimeError(f"Periodogram computation failed: {e}")
