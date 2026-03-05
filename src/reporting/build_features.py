# src/reporting/build_features.py

"""
Aggregate rolling-window seasonality data into per-ticker, per-frequency summaries.

Consumes:   df_rolling output from SeasonalityETL.fit_rolling()
            One row per (ticker, interval, freq, window_start)

Produces:   One summary row per (ticker, freq) with:
            - mean, std, latest, trend for each raw metric and meta-score
            - window_count: how many valid windows contributed
            - last_window: date of the most recent window

This module has no dependency on src/pipeline; it only consumes DataFrames,
to make it testable with synthetic datasets
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# column groups are defined once here so downstream modules can import them
# rather than hard coding the same lists in multiple places

# raw individual metrics (pre-aggregation)
RAW_METRIC_COLS = ["acf_lag_val", "p2m_val", "stl_strength"]

# composite scores produced by meta_scores.add_meta_scores()
SCORE_COLS = [
    "seasonality_score_linear",
    "seasonality_score_geom",
    "seasonality_score_harmonic",
]

# all numeric columns we want to summarise
ALL_METRIC_COLS = RAW_METRIC_COLS + SCORE_COLS

# columns that must be present in the input DataFrame
REQUIRED_INPUT_COLS = ["ticker", "interval", "freq", "window_start"] + ALL_METRIC_COLS


def build_features(
    df_rolling: pd.DataFrame,
    last_n_windows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collapse a long-form rolling-window DataFrame into one summary row per
    (ticker, freq) by computing mean, std, latest value, and trend for every
    metric and meta-score column.

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Output of SeasonalityETL.fit_rolling().
        Required columns: ticker, interval, freq, window_start,
        acf_lag_val, p2m_val, stl_strength,
        seasonality_score_linear, seasonality_score_geom,
        seasonality_score_harmonic.

    last_n_windows : int, optional
        If provided, only the most recent N windows per (ticker, freq) are
        used for aggregation. Useful for robustness checks (e.g. last_n_windows=15
        tells you whether seasonality has been consistent recently).
        If None, all available windows are used.

    Returns
    -------
    pd.DataFrame
        One row per (ticker, freq) with columns:
        - ticker, freq
        - window_count          : number of windows used in aggregation
        - last_window           : date of the most recent window
        - {metric}_mean         : mean across windows
        - {metric}_std          : std across windows (measure of consistency)
        - {metric}_latest       : value in the most recent window
        - {metric}_trend        : latest minus earliest (positive = improving)
        for each metric in ALL_METRIC_COLS.

    Raises
    ------
    ValueError
        If required columns are missing from df_rolling.

    Notes
    -----
    - `std` is the primary robustness indicator: low std means the metric has
      been stable across windows (seasonality is consistent, not a one-off spike).
    - `trend` is the delta from the first to the last window in the selected
      range. A positive trend on harmonic score suggests seasonality is
      strengthening over time.
    - `interval` is intentionally dropped from the output grouping because
      fit_rolling() already slices by interval internally; in practice a single
      interval (e.g. "1d") is used per run. If multi-interval runs are needed,
      re-add `interval` to the groupby keys.
    """
    _validate_input(df_rolling)

    if df_rolling.empty:
        logger.warning("build_features received an empty DataFrame, returning empty result.")
        return pd.DataFrame()

    # ensure window_start is datetime so sorting and .last() work correctly
    df = df_rolling.copy()
    df["window_start"] = pd.to_datetime(df["window_start"])

    # optionally restrict to the N most recent windows per (ticker, freq).
    # --> we sort descending, take the first N rows, then re-sort ascending so
    # --> that "earliest" and "latest" are correctly identified later.
    if last_n_windows is not None:
        if not isinstance(last_n_windows, int) or last_n_windows < 1:
            raise ValueError(f"last_n_windows must be a positive integer, got {last_n_windows}")
        df = (
            df.sort_values("window_start", ascending=False)
            .groupby(["ticker", "freq"], group_keys=False)
            .head(last_n_windows)
        )
        logger.debug(f"Restricted to last {last_n_windows} windows per (ticker, freq).")

    # sort ascending so that .first() = earliest window, .last() = latest window
    df = df.sort_values("window_start", ascending=True)

    summary_rows = []

    for (ticker, freq), group in df.groupby(["ticker", "freq"]):
        row: dict = {
            "ticker": ticker,
            "freq": freq,
            # how many windows contributed? important for interpreting std
            "window_count": len(group),
            # most recent window date --> useful for staleness checks in the report
            "last_window": group["window_start"].iloc[-1],
        }

        for col in ALL_METRIC_COLS:
            values = group[col]

            # mean: central tendency across windows
            row[f"{col}_mean"] = values.mean()

            # std: spread across windows --> low = consistent, high = volatile
            # ddof=0 (population std) because we're describing this specific
            # window sample, not inferring a population
            row[f"{col}_std"] = values.std(ddof=0)

            # latest: the most recent observed value
            row[f"{col}_latest"] = values.iloc[-1]

            # trend: direction of change from first to last window.
            # positive = metric improved over time; negative = deteriorated.
            row[f"{col}_trend"] = float(values.iloc[-1]) - float(values.iloc[0])

        summary_rows.append(row)

    df_features = pd.DataFrame(summary_rows)

    logger.info(
        f"build_features: produced {len(df_features)} summary rows "
        f"from {len(df_rolling)} rolling windows "
        f"({df_features['ticker'].nunique()} tickers, "
        f"{df_features['freq'].nunique()} freq buckets)."
    )

    return df_features


def _validate_input(df: pd.DataFrame) -> None:
    """
    Check that all required columns are present in the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The rolling-window DataFrame to validate.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in REQUIRED_INPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"build_features: input DataFrame is missing required columns: {missing}. "
            f"Expected all of: {REQUIRED_INPUT_COLS}"
        )
