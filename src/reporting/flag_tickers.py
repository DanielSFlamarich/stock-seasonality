# src/reporting/flag_tickers.py

"""
Classify tickers by seasonality strength and compute universe-relative percentile ranks.

Consumes:   df_features output from build_features.build_features()
            One row per (ticker, freq)

Produces:   The same DataFrame with added columns:
            - stl_available         : bool — False when stl_strength_mean is NaN
            - low_window_count      : bool — advisory; True when window_count is below
                                      MIN_RELIABLE_WINDOW_COUNT. Flags are still computed
                                      for these tickers but carry less statistical weight.
                                      Surfaced in the report as supplementary context, not
                                      as a disqualifier.
            - {metric}_mean_pct     : float [0, 100] — percentile rank within the same
                                      freq bucket (see NaN policy below)
            - superperformer_flag   : bool — ticker clears all raw metric and harmonic
                                      score thresholds and has STL confirmed

Design note: no_seasonality_flag was deliberately omitted. The superperformer_flag is a
binary gate, a ticker either clears all thresholds or it does not.

------------------------------------------------------------------------------------------
STL decomposition legitimately returns NaN for tickers with short or ME-frequency windows
(insufficient observations for the required seasonal period). This is *informative*, not a
data quality error.

Consequences enforced here:
  1. `stl_available` is set to False when `stl_strength_mean` is NaN.
  2. STL percentile rank (`stl_strength_mean_pct`) is computed only within the
     subpopulation of tickers where `stl_available=True` for that freq bucket.
     Tickers with stl_available=False receive NaN for this percentile column,
     preserving the distinction between "scored poorly" and "could not be scored".
  3. `superperformer_flag` requires stl_available=True. A ticker cannot be
     classified as top-tier if STL confirmation is absent, regardless of how
     strong ACF and P2M appear.

Rationale: imputing or substituting STL NaNs would inject false confidence into the
flag logic. Explicit unavailability is more honest and easier to audit downstream.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from reporting.build_features import (  # noqa: E402 : adjust to package path as needed
    ALL_METRIC_COLS,
    RAW_METRIC_COLS,
)

logger = logging.getLogger(__name__)


# Threshold constants — all in one place so callers can override by import

# to qualify as a superperformer a ticker must exceed ALL of these on _mean:
SUPERPERFORMER_RAW_THRESHOLD: float = 0.50  # each raw metric individually
SUPERPERFORMER_HARMONIC_THRESHOLD: float = 0.60  # most conservative aggregate score

# percentile rank floor to be considered "top-tier" (informational, not used in flags)
SUPERPERFORMER_PERCENTILE_FLOOR: float = 80.0

# minimum window_count to trust the aggregated features for flag purposes.
# tickers with fewer windows will still appear in the output but their flags
# are less reliable; callers should inspect window_count alongside flags.
MIN_RELIABLE_WINDOW_COUNT: int = 5

# columns used in flag decisions; declared explicitly so downstream imports
# can reference them without re-deriving from ALL_METRIC_COLS.
SUPERPERFORMER_RAW_METRICS: list[str] = [f"{m}_mean" for m in RAW_METRIC_COLS]
SUPERPERFORMER_SCORE_METRIC: str = "seasonality_score_harmonic_mean"

# required columns in df_features input
REQUIRED_FEATURE_COLS: list[str] = ["ticker", "freq", "window_count"] + [
    f"{m}_mean" for m in ALL_METRIC_COLS
]

# Public API


def flag_tickers(
    df_features: pd.DataFrame,
    superperformer_raw_threshold: float = SUPERPERFORMER_RAW_THRESHOLD,
    superperformer_harmonic_threshold: float = SUPERPERFORMER_HARMONIC_THRESHOLD,
    min_reliable_windows: Optional[int] = MIN_RELIABLE_WINDOW_COUNT,
) -> pd.DataFrame:
    """
    Classify tickers and attach universe-relative percentile ranks.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of build_features.build_features(). One row per (ticker, freq).

    superperformer_raw_threshold : float
        Each individual raw metric mean must exceed this to qualify as a
        superperformer. Default: 0.50.

    superperformer_harmonic_threshold : float
        Harmonic meta-score mean must exceed this to qualify as a superperformer.
        The harmonic score is the most conservative aggregate — using it as the
        composite gate means a ticker must be strong across all three raw metrics,
        not just on average. Default: 0.60.

    min_reliable_windows : int, optional
        Tickers with fewer windows than this get `low_window_count=True`.
        Flags are still computed — this is advisory context for the report,
        not a disqualifier. Set to None to disable. Default: 5.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns (in order):
        - stl_available             : bool
        - low_window_count          : bool (if min_reliable_windows is set)
        - {metric}_mean_pct         : float [0, 100] for each metric in ALL_METRIC_COLS
        - superperformer_flag       : bool

    Raises
    ------
    ValueError
        If required columns are missing from df_features.

    Notes
    -----
    - Percentile ranks are computed within freq buckets, not globally.
      A ticker ranked 95th in "ME" may rank differently in "YE" — seasonal
      strength is not comparable across frequencies.
    - The `percentileofscore` kind is "rank" (average of lower and upper).
    - For single-ticker freq buckets the percentile rank is 100.0 by definition.
      Mathematically correct but uninformative — use absolute metric values in
      those cases.
    """
    _validate_features_input(df_features)

    if df_features.empty:
        logger.warning("flag_tickers received an empty DataFrame; returning empty result.")
        return df_features.copy()

    df = df_features.copy()

    # 1. STL availability gate  (Option A, see module docstring)
    df["stl_available"] = df["stl_strength_mean"].notna()

    n_unavailable = (~df["stl_available"]).sum()
    if n_unavailable > 0:
        logger.info(
            f"flag_tickers: {n_unavailable}/{len(df)} tickers have stl_available=False "
            f"(STL NaN — insufficient observations for seasonal period). "
            f"These tickers are excluded from the STL percentile rank and cannot "
            f"qualify as superperformers. See module docstring for full rationale."
        )

    # 2. low-window-count advisory flag
    if min_reliable_windows is not None:
        df["low_window_count"] = df["window_count"] < min_reliable_windows
        n_low = df["low_window_count"].sum()
        if n_low > 0:
            logger.warning(
                f"flag_tickers: {n_low} tickers have window_count < {min_reliable_windows}. "
                f"Their flags are computed but less reliable. "
                f"Inspect `window_count` alongside flags in downstream reports."
            )

    # 3. Percentile ranks — within freq bucket
    for metric in ALL_METRIC_COLS:
        col = f"{metric}_mean"
        pct_col = f"{metric}_mean_pct"

        if metric == "stl_strength":
            # STL percentile is computed only within the stl_available subpopulation
            # per freq bucket. NaN rows receive NaN (not 0) preserving the
            # distinction between "scored poorly" and "could not be scored".
            df[pct_col] = _percentile_rank_with_stl_gate(df, col)
        else:
            df[pct_col] = df.groupby("freq")[col].transform(
                lambda s: s.apply(
                    lambda v: (
                        percentileofscore(s.dropna(), v, kind="rank") if pd.notna(v) else np.nan
                    )
                )
            )

    # 4. Superperformer flag
    # requires:
    #   a) stl_available=True   = all three raw metrics must be confirmed
    #   b) each raw metric mean > superperformer_raw_threshold
    #   c) harmonic score mean  > superperformer_harmonic_threshold
    #
    # rationale for (a): a superperformer badge should be unambiguous.
    # if STL could not confirm seasonality (NaN), we cannot award top-tier
    # status regardless of how strong ACF and P2M appear.
    raw_conditions = pd.concat(
        [df[col] > superperformer_raw_threshold for col in SUPERPERFORMER_RAW_METRICS],
        axis=1,
    ).all(axis=1)

    harmonic_condition = df[SUPERPERFORMER_SCORE_METRIC] > superperformer_harmonic_threshold

    df["superperformer_flag"] = df["stl_available"] & raw_conditions & harmonic_condition

    n_super = df["superperformer_flag"].sum()
    logger.info(
        f"flag_tickers: {n_super}/{len(df)} tickers flagged as superperformer "
        f"(stl_available + all raw means > {superperformer_raw_threshold} "
        f"+ harmonic_mean > {superperformer_harmonic_threshold})."
    )

    return df


# Helpers


def _percentile_rank_with_stl_gate(
    df: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    Compute per-freq percentile rank for a column, restricting the reference
    population to rows where stl_available=True.

    Rows with stl_available=False receive NaN (not 0) to preserve the semantic
    distinction between a low score and an unavailable score.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `freq`, `stl_available`, and `col`.
    col : str
        Column to rank (expected: "stl_strength_mean").

    Returns
    -------
    pd.Series
        Percentile rank series aligned to df.index.
    """
    result = pd.Series(np.nan, index=df.index)

    for freq, group in df.groupby("freq"):
        available_mask = group["stl_available"]
        available_group = group.loc[available_mask]

        if available_group.empty:
            logger.debug(
                f"flag_tickers: no stl_available tickers in freq={freq}; "
                f"stl_strength_mean_pct will be NaN for all tickers in this freq bucket."
            )
            continue

        ref_scores = available_group[col].dropna().values

        for idx, row in available_group.iterrows():
            v = row[col]
            if pd.notna(v):
                result.loc[idx] = percentileofscore(ref_scores, v, kind="rank")
            # else: stays NaN

    return result


def _validate_features_input(df: pd.DataFrame) -> None:
    """
    Raise ValueError if required columns are missing from df_features.

    Parameters
    ----------
    df : pd.DataFrame
        The features DataFrame to validate.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in REQUIRED_FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"flag_tickers: df_features is missing required columns: {missing}. "
            f"Expected all of: {REQUIRED_FEATURE_COLS}"
        )
