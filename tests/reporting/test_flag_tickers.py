# tests/reporting/test_flag_tickers.py

"""
Unit tests for flag_tickers.flag_tickers().

All tests use synthetic DataFrames that mimic build_features output.
No yfinance or SeasonalityETL dependency — fast and deterministic.

Fixture strategy
----------------
make_row()        : helper that builds one df_features row from readable scalar inputs
df_three_tickers  : 3 tickers / 1 freq (ME): PERFECT, MID, FLAT(stl=NaN)
df_multi_freq     : same tickers across ME and YE to test per-freq isolation
df_all_stl_nan    : all tickers in a single freq have stl NaN (edge case)
"""

import numpy as np
import pandas as pd
import pytest

from reporting.build_features import ALL_METRIC_COLS
from reporting.flag_tickers import (
    MIN_RELIABLE_WINDOW_COUNT,
    SUPERPERFORMER_RAW_THRESHOLD,
    flag_tickers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _harmonic(acf: float, p2m: float, stl: float, eps: float = 1e-6) -> float:
    """
    Compute harmonic mean of three values (mirrors meta_scores logic).
    """
    return 3 / (1 / max(acf, eps) + 1 / max(p2m, eps) + 1 / max(stl, eps))


def make_row(
    ticker: str,
    acf: float,
    p2m: float,
    stl: float | None,  # None --> STL unavailable (NaN)
    freq: str = "ME",
    window_count: int = 12,
) -> dict:
    """
    Build one df_features row from scalar metric values.

    Derived aggregate columns (linear, geom, harmonic) are computed
    consistently from the three raw inputs so tests don't encode
    magic numbers that drift from the real computation.
    """
    stl_v = np.nan if stl is None else stl
    if stl is not None:
        harmonic = _harmonic(acf, p2m, stl)
        linear = float(np.mean([acf, p2m, stl]))
        geom = float(np.exp(np.mean(np.log([max(acf, 1e-6), max(p2m, 1e-6), max(stl, 1e-6)]))))
    else:
        # STL unavailable: pipeline would produce low/nan composites;
        # we use a plausible but clearly weak value for test determinism.
        harmonic = 0.05
        linear = float(np.mean([acf, p2m]))
        geom = 0.05

    row: dict = {
        "ticker": ticker,
        "freq": freq,
        "window_count": window_count,
        "last_window": "2024-12-01",
        # _mean columns (used by flag logic)
        "acf_lag_val_mean": acf,
        "p2m_val_mean": p2m,
        "stl_strength_mean": stl_v,
        "seasonality_score_linear_mean": linear,
        "seasonality_score_geom_mean": geom,
        "seasonality_score_harmonic_mean": harmonic,
    }

    # remaining _std / _latest / _trend columns are required by _validate_input
    # but not used in flag logic; fill with sensible stubs
    for metric in ALL_METRIC_COLS:
        is_stl = metric == "stl_strength"
        stub = np.nan if (is_stl and stl is None) else 0.01
        row[f"{metric}_std"] = stub
        row[f"{metric}_latest"] = row[f"{metric}_mean"]
        row[f"{metric}_trend"] = stub

    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_three_tickers() -> pd.DataFrame:
    """
    Three tickers in a single ME freq bucket.

    PERFECT.SYN : strong signal, stl_available=True  --> should be superperformer
    MID.SYN     : moderate signal, stl_available=True --> should NOT be superperformer
    FLAT.SYN    : weak signal,  stl_available=False   --> cannot be superperformer
    """
    return pd.DataFrame(
        [
            make_row("PERFECT.SYN", acf=0.90, p2m=0.85, stl=0.88),
            make_row("MID.SYN", acf=0.45, p2m=0.50, stl=0.48),
            make_row("FLAT.SYN", acf=0.05, p2m=0.08, stl=None),
        ]
    )


@pytest.fixture
def df_multi_freq() -> pd.DataFrame:
    """
    Two freq buckets (ME, YE) each with a strong and a weak ticker.
    Used to verify that percentile ranks are computed independently per freq.
    """
    return pd.DataFrame(
        [
            make_row("STRONG.ME", acf=0.90, p2m=0.85, stl=0.88, freq="ME"),
            make_row("WEAK.ME", acf=0.10, p2m=0.12, stl=0.11, freq="ME"),
            make_row("STRONG.YE", acf=0.80, p2m=0.75, stl=0.82, freq="YE"),
            make_row("WEAK.YE", acf=0.15, p2m=0.18, stl=0.16, freq="YE"),
        ]
    )


@pytest.fixture
def df_all_stl_nan() -> pd.DataFrame:
    """
    All tickers in the freq bucket have STL unavailable.
    """
    return pd.DataFrame(
        [
            make_row("A.SYN", acf=0.80, p2m=0.75, stl=None),
            make_row("B.SYN", acf=0.40, p2m=0.35, stl=None),
        ]
    )


@pytest.fixture
def df_single_ticker() -> pd.DataFrame:
    """
    Single ticker in a freq bucket — percentile rank edge case.
    """
    return pd.DataFrame(
        [
            make_row("ONLY.SYN", acf=0.70, p2m=0.65, stl=0.68),
        ]
    )


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


def test_missing_required_column_raises():
    """
    _validate_features_input must raise ValueError for missing columns.
    """
    df = pd.DataFrame([make_row("X.SYN", 0.5, 0.5, 0.5)])
    df = df.drop(columns=["acf_lag_val_mean"])
    with pytest.raises(ValueError, match="missing required columns"):
        flag_tickers(df)


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------


def test_empty_dataframe_returns_empty():
    """
    Empty input must return empty without error.
    """
    df_empty = pd.DataFrame(columns=pd.DataFrame([make_row("X.SYN", 0.5, 0.5, 0.5)]).columns)
    result = flag_tickers(df_empty)
    assert result.empty


# ---------------------------------------------------------------------------
# 3–4. stl_available derivation
# ---------------------------------------------------------------------------


def test_stl_available_true_when_stl_not_nan(df_three_tickers):
    result = flag_tickers(df_three_tickers)
    perfect = result.loc[result.ticker == "PERFECT.SYN"].iloc[0]
    assert perfect["stl_available"] is True or perfect["stl_available"]


def test_stl_available_false_when_stl_nan(df_three_tickers):
    result = flag_tickers(df_three_tickers)
    flat = result.loc[result.ticker == "FLAT.SYN"].iloc[0]
    assert flat["stl_available"] is False or not flat["stl_available"]


# ---------------------------------------------------------------------------
# 5–6. STL percentile rank — NaN policy (Option A)
# ---------------------------------------------------------------------------


def test_stl_nan_ticker_gets_nan_percentile_not_zero(df_three_tickers):
    """
    stl_available=False tickers must receive NaN for stl_strength_mean_pct.
    Receiving 0 would be incorrect as it implies "scored at the bottom" rather
    than "could not be scored", which are semantically different outcomes.
    """
    result = flag_tickers(df_three_tickers)
    flat = result.loc[result.ticker == "FLAT.SYN"].iloc[0]
    assert np.isnan(
        flat["stl_strength_mean_pct"]
    ), "stl_strength_mean_pct should be NaN for stl_available=False, not 0"


def test_stl_available_ticker_gets_valid_percentile(df_three_tickers):
    result = flag_tickers(df_three_tickers)
    for ticker in ["PERFECT.SYN", "MID.SYN"]:
        pct = result.loc[result.ticker == ticker, "stl_strength_mean_pct"].iloc[0]
        assert 0.0 <= pct <= 100.0, f"{ticker} stl pct should be in [0, 100], got {pct}"


# ---------------------------------------------------------------------------
# 7. All STL NaN in a freq bucket — no crash, all pct NaN
# ---------------------------------------------------------------------------


def test_all_stl_nan_in_freq_bucket_no_crash(df_all_stl_nan):
    """
    When every ticker in a freq bucket has stl_available=False,
    stl_strength_mean_pct must be NaN for all --> no crash, no silent 0.
    """
    result = flag_tickers(df_all_stl_nan)
    assert (
        result["stl_strength_mean_pct"].isna().all()
    ), "All stl pcts should be NaN when no ticker in the bucket has STL available"


# ---------------------------------------------------------------------------
# 8. Percentile ranks are per-freq, not global
# ---------------------------------------------------------------------------


def test_percentile_ranks_are_per_freq_not_global(df_multi_freq):
    """
    Both STRONG.ME and STRONG.YE should rank 100th in their respective freq
    buckets. If ranking were global, the weaker strong ticker would rank lower.
    """
    result = flag_tickers(df_multi_freq)
    for ticker in ["STRONG.ME", "STRONG.YE"]:
        pct = result.loc[result.ticker == ticker, "acf_lag_val_mean_pct"].iloc[0]
        assert pct == pytest.approx(
            100.0
        ), f"{ticker} should rank 100th within its own freq bucket, got {pct}"


# ---------------------------------------------------------------------------
# 9. Single-ticker freq bucket → percentile = 100.0
# ---------------------------------------------------------------------------


def test_single_ticker_freq_bucket_percentile_is_100(df_single_ticker):
    """
    With only one ticker in a freq bucket, percentileofscore returns 100.0
    by definition (the score is >= 100% of the population including itself).
    This is mathematically correct but informationally limited, window_count
    and absolute metric values should guide interpretation in this case.
    """
    result = flag_tickers(df_single_ticker)
    pct = result["acf_lag_val_mean_pct"].iloc[0]
    assert pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 10. Superperformer requires stl_available=True (Option A)
# ---------------------------------------------------------------------------


def test_superperformer_requires_stl_available(df_three_tickers):
    """
    A ticker with STL NaN cannot be a superperformer even if ACF and P2M are
    strong. This is Option A: STL absence is informative, not ignorable.
    """
    # Inject a high-ACF/P2M ticker with no STL
    df = pd.DataFrame(
        [
            make_row("STRONG_NO_STL.SYN", acf=0.95, p2m=0.92, stl=None),
        ]
    )
    result = flag_tickers(df)
    assert not result["superperformer_flag"].iloc[0]


# ---------------------------------------------------------------------------
# 11. Superperformer requires ALL raw metrics above threshold
# ---------------------------------------------------------------------------


def test_superperformer_fails_if_one_raw_metric_below_threshold():
    """
    One weak raw metric must block superperformer even if the others are strong.
    """
    df = pd.DataFrame(
        [
            # p2m is well below SUPERPERFORMER_RAW_THRESHOLD
            make_row("WEAK_P2M.SYN", acf=0.90, p2m=0.10, stl=0.88),
        ]
    )
    result = flag_tickers(df)
    assert not result["superperformer_flag"].iloc[0]


# ---------------------------------------------------------------------------
# 12. Exactly at threshold → NOT superperformer (strict >)
# ---------------------------------------------------------------------------


def test_superperformer_threshold_is_strict_greater_than():
    """
    Threshold comparisons use strict >, so a ticker sitting exactly at
    SUPERPERFORMER_RAW_THRESHOLD must not be flagged.
    """
    t = SUPERPERFORMER_RAW_THRESHOLD  # e.g. 0.50
    df = pd.DataFrame(
        [
            make_row("EXACT.SYN", acf=t, p2m=t, stl=t),
        ]
    )
    result = flag_tickers(df)
    assert not result["superperformer_flag"].iloc[
        0
    ], f"Ticker at exactly threshold={t} should not qualify (strict >)"


# ---------------------------------------------------------------------------
# 13. Superperformer requires harmonic score above threshold
# ---------------------------------------------------------------------------


def test_superperformer_fails_if_harmonic_below_threshold():
    """
    Raw metrics can pass while the harmonic composite fails
    the harmonic gate must independently block the flag.
    """
    # Use custom thresholds to isolate the harmonic gate:
    # raw threshold is 0.0 (always passes), harmonic threshold is high.
    df = pd.DataFrame(
        [
            make_row("LOW_HARMONIC.SYN", acf=0.55, p2m=0.55, stl=0.55),
        ]
    )
    result = flag_tickers(
        df,
        superperformer_raw_threshold=0.0,  # raw gate: always passes
        superperformer_harmonic_threshold=0.99,  # harmonic gate: impossible to reach
    )
    assert not result["superperformer_flag"].iloc[0]


# ---------------------------------------------------------------------------
# 14. Happy path — all conditions met
# ---------------------------------------------------------------------------


def test_superperformer_happy_path(df_three_tickers):
    """
    PERFECT.SYN clears all gates and must be flagged as superperformer.
    """
    result = flag_tickers(df_three_tickers)
    perfect = result.loc[result.ticker == "PERFECT.SYN"].iloc[0]
    assert perfect["superperformer_flag"]


def test_mid_ticker_is_not_superperformer(df_three_tickers):
    """
    MID.SYN is above noise but below superperformer thresholds.
    """
    result = flag_tickers(df_three_tickers)
    mid = result.loc[result.ticker == "MID.SYN"].iloc[0]
    assert not mid["superperformer_flag"]


# ---------------------------------------------------------------------------
# 15. Custom thresholds are respected
# ---------------------------------------------------------------------------


def test_custom_thresholds_change_superperformer_outcome():
    """
    A MID-level ticker should become a superperformer when thresholds are
    lowered, and stop being one when they are raised.
    """
    df = pd.DataFrame([make_row("MID.SYN", acf=0.45, p2m=0.50, stl=0.48)])

    result_low = flag_tickers(
        df, superperformer_raw_threshold=0.30, superperformer_harmonic_threshold=0.30
    )
    assert result_low["superperformer_flag"].iloc[0], "Should qualify with lowered thresholds"

    result_high = flag_tickers(
        df, superperformer_raw_threshold=0.80, superperformer_harmonic_threshold=0.80
    )
    assert not result_high["superperformer_flag"].iloc[
        0
    ], "Should not qualify with raised thresholds"


# ---------------------------------------------------------------------------
# 16. low_window_count advisory flag
# ---------------------------------------------------------------------------


def test_low_window_count_flagged_below_threshold():
    df = pd.DataFrame(
        [
            make_row("SPARSE.SYN", acf=0.90, p2m=0.85, stl=0.88, window_count=2),
        ]
    )
    result = flag_tickers(df)
    assert result["low_window_count"].iloc[0]


def test_sufficient_window_count_not_flagged():
    df = pd.DataFrame(
        [
            make_row(
                "DENSE.SYN",
                acf=0.90,
                p2m=0.85,
                stl=0.88,
                window_count=MIN_RELIABLE_WINDOW_COUNT,
            ),
        ]
    )
    result = flag_tickers(df)
    assert not result["low_window_count"].iloc[0]


# ---------------------------------------------------------------------------
# 17. low_window_count does NOT disqualify superperformer
# ---------------------------------------------------------------------------


def test_low_window_count_does_not_block_superperformer():
    """
    low_window_count is advisory context for the report reader, not a gate.
    A ticker with few but strong windows should still be flagged as superperformer.
    """
    df = pd.DataFrame(
        [
            make_row("SPARSE_STRONG.SYN", acf=0.90, p2m=0.85, stl=0.88, window_count=2),
        ]
    )
    result = flag_tickers(df)
    assert result["low_window_count"].iloc[0]
    assert result["superperformer_flag"].iloc[0]


# ---------------------------------------------------------------------------
# 18. min_reliable_windows=None disables low_window_count column
# ---------------------------------------------------------------------------


def test_min_reliable_windows_none_omits_column():
    df = pd.DataFrame([make_row("X.SYN", 0.5, 0.5, 0.5, window_count=1)])
    result = flag_tickers(df, min_reliable_windows=None)
    assert "low_window_count" not in result.columns
