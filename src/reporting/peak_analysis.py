# src/reporting/peak_analysis.py

"""
Peak detection and inter-peak gap analysis on raw price series.

Consumes:   df_prices : standard DataLoader / synthetic_data_generator output.
            Required columns: date (or DatetimeIndex), ticker, close.
            Expected interval: daily ('1d'). Peak detection is performed on the
            full-granularity daily series regardless of the target freq; freq only
            controls the minimum spacing between accepted peaks (see below).

Produces:   One summary row per (ticker, freq) with:
            - peak_count          : int   --> number of peaks detected
            - mean_peak_gap_days  : float --> mean calendar days between consecutive peaks
            - std_peak_gap_days   : float --> std of gap (ddof=0); low = stable cadence,
                                            high = rhythm is sliding across the calendar

Output joins onto df_features on (ticker, freq) to feed report_generator.

Design decisions (documented here to explain the choices to future readers)
---------------------------------------------------------------------------
1. Inter-peak gap in calendar days (not trading days).
   Rationale: no market-calendar dependency, consistent across instruments,
   and calendar days are the natural unit when asking "does this peak in January
   every year?" Chosen over calendar-position std (day-of-month / day-of-week),
   which would miss cases where seasonality slides but maintains a steady rhythm.

2. scipy.signal.find_peaks with prominence threshold.
   Prominence = prominence_factor × IQR(close). IQR-based scaling makes the
   threshold scale-invariant across instruments at different price levels (a $5
   stock and a $500 stock use proportionally equivalent thresholds). Alternative
   of a fixed absolute threshold would require per-ticker calibration.

3. Minimum peak distance derived from freq (in trading-day samples, daily series):
       W  --> 3   (half a week)
       ME --> 15  (half a month)
       QE --> 45  (half a quarter)
       YE --> 126 (half a year in trading days)
   This prevents detecting intra-period noise as separate seasonal peaks while
   still allowing one peak per period. Expressed in samples (not calendar days)
   because find_peaks operates on array indices.

4. < 2 detected peaks → NaN for gap stats.
   One peak produces zero gaps; returning 0 would be misleading. NaN correctly
   signals "insufficient peaks to measure a rhythm" to downstream consumers.

5. ddof=0 for std_peak_gap_days — population std over observed gaps, consistent
   with the ddof=0 choice made in build_features.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# freq --> minimum peak distance in samples (daily series)
# see design decision 3 in module docstring.
# ---------------------------------------------------------------------------
FREQ_MIN_DISTANCE: dict[str, int] = {
    "W": 3,  # half a week
    "ME": 15,  # half a month
    "QE": 45,  # half a quarter
    "YE": 126,  # half a year (trading days)
}

# default IQR multiplier for prominence threshold (design decision 2).
DEFAULT_PROMINENCE_FACTOR: float = 0.1

# required columns in df_prices input.
REQUIRED_PRICE_COLS: list[str] = ["ticker", "close"]

# minimum peaks required to compute gap statistics (design decision 4).
MIN_PEAKS_FOR_GAPS: int = 2


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def summarise_peaks(
    df_prices: pd.DataFrame,
    freqs: Optional[list[str]] = None,
    prominence_factor: float = DEFAULT_PROMINENCE_FACTOR,
) -> pd.DataFrame:
    """
    Compute inter-peak gap statistics for every (ticker, freq) combination.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Raw daily price data. Required columns: ticker, close, and either a
        'date' column or a DatetimeIndex. Expected interval: daily ('1d').

    freqs : list of str, optional
        Seasonality frequencies to analyse. Each freq controls the minimum
        spacing between accepted peaks. Defaults to all supported freqs:
        ['W', 'ME', 'QE', 'YE'].

    prominence_factor : float
        Prominence threshold = prominence_factor × IQR(close). Default: 0.1.
        Increase to detect only major peaks; decrease to be more sensitive.

    Returns
    -------
    pd.DataFrame
        One row per (ticker, freq) with columns:
        - ticker
        - freq
        - peak_count          : int
        - mean_peak_gap_days  : float (NaN if peak_count < 2)
        - std_peak_gap_days   : float (NaN if peak_count < 2)

    Raises
    ------
    ValueError
        If required columns are missing from df_prices.
    """
    _validate_prices_input(df_prices)

    if freqs is None:
        freqs = list(FREQ_MIN_DISTANCE.keys())

    unknown = [f for f in freqs if f not in FREQ_MIN_DISTANCE]
    if unknown:
        raise ValueError(
            f"summarise_peaks: unsupported freq(s) {unknown}. "
            f"Supported: {list(FREQ_MIN_DISTANCE.keys())}"
        )

    # normalise date column to DatetimeIndex
    df = df_prices.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    rows = []
    for ticker, group in df.groupby("ticker"):
        series = group["close"].sort_index().dropna()

        if series.empty:
            logger.warning(f"peak_analysis: no close data for {ticker}, skipping.")
            continue

        for freq in freqs:
            stats = compute_peak_stats(series, freq=freq, prominence_factor=prominence_factor)
            rows.append({"ticker": ticker, "freq": freq, **stats})

    df_peaks = pd.DataFrame(rows)

    if df_peaks.empty:  # dodge a KeyError if df with valid columns but zero matching rows
        return df_peaks

    logger.info(
        f"summarise_peaks: produced {len(df_peaks)} rows "
        f"({df_peaks['ticker'].nunique()} tickers × {len(freqs)} freqs)."
    )

    return df_peaks


def compute_peak_stats(
    series: pd.Series,
    freq: str,
    prominence_factor: float = DEFAULT_PROMINENCE_FACTOR,
) -> dict:
    """
    Detect peaks in a single daily price series and compute gap statistics.

    Parameters
    ----------
    series : pd.Series
        Close prices with a DatetimeIndex, sorted ascending, no NaNs.

    freq : str
        Target seasonality frequency. Controls minimum peak spacing.
        One of: 'W', 'ME', 'QE', 'YE'.

    prominence_factor : float
        Prominence = prominence_factor × IQR(series). Default: 0.1.

    Returns
    -------
    dict with keys:
        peak_count          : int
        mean_peak_gap_days  : float or NaN
        std_peak_gap_days   : float or NaN
    """
    if freq not in FREQ_MIN_DISTANCE:
        raise ValueError(
            f"compute_peak_stats: unsupported freq '{freq}'. "
            f"Supported: {list(FREQ_MIN_DISTANCE.keys())}"
        )

    values = series.to_numpy(dtype=float)
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))

    # flat or near-flat series: no meaningful peaks
    if iqr == 0:
        logger.debug("compute_peak_stats: IQR=0 for series, returning zero peaks.")
        return _empty_stats()

    prominence = prominence_factor * iqr
    min_distance = FREQ_MIN_DISTANCE[freq]

    peak_indices, _ = find_peaks(values, prominence=prominence, distance=min_distance)
    peak_count = len(peak_indices)

    if peak_count < MIN_PEAKS_FOR_GAPS:
        logger.debug(
            f"compute_peak_stats: only {peak_count} peak(s) detected for freq={freq}; "
            f"need >= {MIN_PEAKS_FOR_GAPS} to compute gap stats."
        )
        return {
            "peak_count": peak_count,
            "mean_peak_gap_days": np.nan,
            "std_peak_gap_days": np.nan,
        }

    peak_dates = series.index[peak_indices]
    gaps_days = pd.Series(peak_dates).diff().dropna().dt.days.to_numpy(dtype=float)

    return {
        "peak_count": peak_count,
        "mean_peak_gap_days": float(np.mean(gaps_days)),
        # ddof=0: population std over observed gaps — consistent with build_features
        "std_peak_gap_days": float(np.std(gaps_days, ddof=0)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_stats() -> dict:
    """Return a NaN-filled stats dict for series that yield no usable peaks."""
    return {
        "peak_count": 0,
        "mean_peak_gap_days": np.nan,
        "std_peak_gap_days": np.nan,
    }


def _validate_prices_input(df: pd.DataFrame) -> None:
    """
    Raise ValueError if required columns are missing from df_prices.

    Parameters
    ----------
    df : pd.DataFrame

    Raises
    ------
    ValueError
    """
    missing = [c for c in REQUIRED_PRICE_COLS if c not in df.columns and c != "date"]
    # 'date' can be a column or the index — accept either
    has_date = "date" in df.columns or isinstance(df.index, pd.DatetimeIndex)
    if not has_date:
        missing.append("date (column or DatetimeIndex)")
    if missing:
        raise ValueError(
            f"summarise_peaks: df_prices is missing required columns: {missing}. "
            f"Expected: ticker, close, and date (column or DatetimeIndex)."
        )
